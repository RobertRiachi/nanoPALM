import os
import torch
import torch.nn.functional as F
import numpy as np
from math import floor, log
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model import PaLMConfig, PaLM, LayerNorm
from transformers import AutoTokenizer
from tqdm import tqdm

# TODO: clean this up
device = "cuda" if torch.cuda.is_available() else "cpu" # No love for MPS for now

datasets_dir = 'data'
dataset = "openwebtext"
grad_accumulation_steps = 4
batch_size = 16 # Paper follows get_bs function defined below, but this might be too extreme for consumer GPUs
block_size = 512 # Paper uses 2048 but this might be a bit too extreme for consumer GPUs

start_iter = 1 #TODO: Update this when loading from checkpoint in the future
max_iters = 10
learning_rate = 3e-4
beta1 = 0.9
beta2 = 1.0 - (start_iter)**(-0.8) # Only for init, dynamically modified during training
weight_decay = learning_rate**2.0 # Only for init, dynamically modified during training
grad_clip = 1.0

eval_only = False

def get_lr(step):
    # 10**-2 for the first 10k steps
    if step < 10000:
        return 10**-2
    # After 10k steps decay by 1/root(steps)
    return step**-0.5 

def get_bs(step):
    # PaLM paper claims 512 bs until 50k steps, 1024 bs until 115k steps, and 2048 bs until complete at 225k
    if step < 50000:
        return 512
    elif step < 115000:
        return 1024
    else:
        return 2024

def update_optim(optim, step):

    for group in optim.param_groups:
        lr = get_lr(step)
        group['lr'] = lr
        group['betas'] = (beta1, 1.0 - (step)**(-0.8))

        # If not in no_decay group update decay
        if group['weight_decay'] != 0.0:
            group['weight_decay'] = lr**2.0

def num_model_params(model):
    units = ['', 'K', 'M', 'B', 'T']
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mag = int(floor(log(total_params, 1000)))
    return f"{int(total_params / 1000**mag)}{units[mag]}"

def load_batch(split, batch_size, device):
    ix = torch.randint(len(split) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((split[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((split[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    
    # Load data & tokenizer
    data_dir = os.path.join(datasets_dir, dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Load model
    config = PaLMConfig(n_embed=512,
                        n_head=4,
                        dropout=0.1,
                        vocab_size=tokenizer.vocab_size,
                        n_layer=2)
    
    model = PaLM(config).to(device)
    num_params = num_model_params(model)
    print(f"Initializing PaLM model with {num_params} params")

    # Disable weight decay for unwanted modules
    # PaLM model has no bias so only include weight params
    # Exclude ln_vocab.weight as it's weight is tied to the word embedding weights
    no_decay_modules = [LayerNorm, torch.nn.Embedding]
    decay_modules = [torch.nn.Linear]
    param_dict = {pn: p for pn, p in model.named_parameters()}
    no_decay_params = [f"{n}.weight" for n, m in model.named_modules() if any(nd for nd in decay_modules if isinstance(m, nd))]
    decay_params = [f"{n}.weight" for n, m in model.named_modules() if any(nd for nd in no_decay_modules if isinstance(m, nd))]

    optimizer_grouped_parameters = [
        {'params': [param_dict[p] for p in decay_params], 'weight_decay': weight_decay},
        {'params': [param_dict[p] for p in no_decay_params if p != 'ln_vocab.weight'], 'weight_decay': 0.0}
    ]

    optim = AdamW(optimizer_grouped_parameters, 
                  lr=learning_rate,
                  betas=(beta1, beta2),
                  fused=True if device == 'cuda' else False)

    # Train
    for step in tqdm(range(start_iter, max_iters + 1)):


        for micro_step in range(grad_accumulation_steps):
        
            # batch_size = get_bs(step) TODO: won't fit on smaller GPUs, figure out work around
            x, y = load_batch(train_data, batch_size, device=device)

            logits = model(x)

            # Paper scales pre-softmax output logits by 1/sqrt(n_embed)
            loss = F.cross_entropy(torch.mul(logits, config.n_embed**-0.5).view(-1, logits.size(-1)),
                                   y.view(-1),
                                   ignore_index=-1)
            loss.backward()

        print(loss)
        # Normalize params by RSM to compensate for lr normalization per param tensor
        for param in model.parameters():
            param.data /= torch.sqrt(torch.mean(torch.square(param.data)))

        # Grad clipping for all models
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        #backward step
        update_optim(optim, step)
        optim.step()
        optim.zero_grad()

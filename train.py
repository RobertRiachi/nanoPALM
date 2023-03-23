import os
import math
import wandb
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from contextlib import nullcontext
from model import PaLMConfig, PaLM, LayerNorm
from transformers import AutoTokenizer
from tqdm import tqdm

# TODO: clean this up
device = "cuda" if torch.cuda.is_available() else "cpu"  # No love for MPS for now
run_name = "palm"

# Evaluation
eval_freq = 100#1000
num_evals = 20#100
best_val_loss = 1e9

# Data
datasets_dir = 'data'
dataset = "openwebtext"
grad_accumulation_steps = 4
batch_size = 16  # Paper follows get_bs function defined below, but this might be too extreme for consumer GPUs
block_size = 512  # Paper uses 2048 but this might be a bit too extreme for consumer GPUs

# Training
# Note: Paper uses lr=1e-2 for 10k iters, then drops to 1/sqrt(step)
# I've found 2e-4 and cosine decay following Chinchilla guidelines to work better
start_iter = 0  # TODO: Update this when loading from checkpoint in the future
max_iters = 100000
warmup_iters = 2000
learning_rate = 2e-4 # Modified at runtime to follow cosine decay
lr_decay_iters = max_iters # Chinchilla
min_learning_rate = learning_rate / 10 # Chinchilla
weight_decay = learning_rate**2.0 # Decoupled weight decay & modified at runtime
grad_clip = 0.5

# Precision
precision = torch.bfloat16
amp_enabled = (precision == torch.bfloat16) # Only works with bfloat16 on my gpu, else loss becomes nan not sure why
amp_ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(enabled=amp_enabled, device_type=device, dtype=precision)
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

# WandB
wandb_logging_enabled = False
wandb_project_name = "nanoPaLM"

# Config
config = {k:v for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}

def get_lr(step):
    # Warmup, else cosine decay learning rate
    if step < warmup_iters:
        return learning_rate * step / warmup_iters
    
    decay = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


def update_optim(optim, step):

    for group in optim.param_groups:
        lr = get_lr(step)
        group['lr'] = lr

        # If not in no_decay group update decay
        if group['weight_decay'] != 0.0:
            group['weight_decay'] = lr**2.0


def num_model_params(model):
    units = ['', 'K', 'M', 'B', 'T']
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    mag = int(math.floor(math.log(total_params, 1000)))
    return f"{int(total_params / 1000**mag)}{units[mag]}"


# Buffers for incoming data
xy = torch.empty((batch_size, block_size+1), dtype=torch.int32).pin_memory()
xy_cuda = torch.empty((batch_size, block_size+1), dtype=torch.int64, device="cuda")

def load_batch(split, batch_size, device):
    global xy
    # Select which items to load
    ix = torch.randint(len(split) - block_size, (batch_size,))
    # Set the relevant elements of xy
    for i, data_i in enumerate(ix):
        xy[i].numpy()[...] = split[data_i:data_i+1+block_size]
    if device == 'cuda':
        # Copy the incoming data directly from pinned memory into cuda mem
        xy_cuda.copy_(xy, non_blocking=True)
        # Slice out x and y
        x = xy_cuda[:, :-1]
        y = xy_cuda[:, 1:]
    else:
        raise NotImplementedError
        #x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate_splits(model, splits, split_names, num_evals, batch_size, device):
    model.eval()
    split_losses = {}
    for split, split_name in zip(splits, split_names):
        losses = torch.zeros(num_evals)
        for i in range(num_evals):
            x, y = load_batch(split, batch_size, device)

            with amp_ctx:
                _, loss = model(x, y)

            losses[i] = loss.item()

        split_losses[split_name] = losses.mean()
    model.train()
    return split_losses


if __name__ == "__main__":

    # Load data & tokenizer
    data_dir = os.path.join(datasets_dir, dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Load model
    palm_config = PaLMConfig(n_embed=768,
                             n_head=6,
                             dropout=0.1,
                             vocab_size=tokenizer.vocab_size,
                             n_layer=4)

    model = PaLM(palm_config).to(device)
    num_params = num_model_params(model)
    print(f"Initializing PaLM model with {num_params} params")

    # Initalize logging
    if wandb_logging_enabled:
        import wandb
        wandb.init(project=wandb_project_name, name=run_name, config=palm_config)

    # Disable weight decay for unwanted modules
    # PaLM model has no bias so only include weight params
    # Exclude ln_vocab.weight as it's weight is tied to the word embedding weights
    no_decay_modules = [LayerNorm, torch.nn.Embedding]
    decay_modules = [torch.nn.Linear]
    param_dict = {pn: p for pn, p in model.named_parameters()}
    no_decay_params = [f"{n}.weight" for n, m in model.named_modules() if any(
        nd for nd in decay_modules if isinstance(m, nd))]
    decay_params = [f"{n}.weight" for n, m in model.named_modules() if any(
        nd for nd in no_decay_modules if isinstance(m, nd))]

    optimizer_grouped_parameters = [
        {'params': [param_dict[p] for p in decay_params], 'weight_decay': weight_decay},
        {'params': [param_dict[p] for p in no_decay_params if p != 'ln_vocab.weight'], 'weight_decay': 0.0}
    ]

    # Model uses betas=(0.9, (1-step**-0.8)), but I've found default works better w/ AdamW
    optim = AdamW(optimizer_grouped_parameters,
                lr=learning_rate,
                fused=True if device == 'cuda' else False)

    model = torch.compile(model)

    # Training loop
    for step in tqdm(range(start_iter, max_iters + 1)):

        update_optim(optim, step)

        if step % eval_freq == 0 and step != 0:
            losses = evaluate_splits(model,
                                     splits=[train_data, val_data],
                                     split_names=['train', 'val'],
                                     num_evals=num_evals,
                                     batch_size=batch_size,
                                     device=device)
            print(f"Step {step}: Training loss={losses['train']}")

            if wandb_logging_enabled:
                wandb.log({
                    "iter": step,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": get_lr(step)
                })

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'model_args': palm_config,
                    'step': step,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint, step:{step}, val_loss:{best_val_loss}")
                check_out = f"checkpoints/{run_name}"

                if not os.path.exists(check_out):
                    os.mkdir(check_out)
                torch.save(checkpoint, os.path.join(check_out, "ckpt.pt"))

        for micro_step in range(grad_accumulation_steps):
            x, y = load_batch(train_data, batch_size, device=device)

            with amp_ctx:
                logits, loss = model(x, y)
            
            scaler.scale(loss).backward()

        # Grad clipping for all model sizes
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optim)
        scaler.update()

        optim.zero_grad()

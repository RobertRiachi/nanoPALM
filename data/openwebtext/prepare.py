# This code comes directly from nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
# However, this might change depending on the future direction of the project, i.e. different tokenization methods, etc...

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

import os
from tqdm import tqdm
import numpy as np
import tiktoken
import multiprocessing as mp
from datasets import load_dataset

NUM_PROC = mp.cpu_count() // 2
ENCODING_METHOD = 'gpt2'

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("openwebtext")

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

enc = tiktoken.get_encoding(ENCODING_METHOD)

def process(example):
    # ignore special tokens and append EOT
    ids = enc.encode_ordinary(example['text']) + [enc.eot_token] 
    return {'ids': ids, 'len': len(ids)}

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=NUM_PROC,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    arr = np.memmap(filename, dtype=np.uint16 , mode='w+', shape=(np.sum(dset['len']),))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()


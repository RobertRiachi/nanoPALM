import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


class LayerNorm(nn.Module):
    # Disable bias in layernorm, since torch doesn't support bias=False
    # From PaLM paper:
    # No biases were used in any of the dense kernels or layer norms.
    # We found this to result in increased training stability for large models.

    def __init__(self, n_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))

    def forward(self, x):
        # None here is for torch functional's bias param
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class MultiQueryAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embed, (config.n_head + 2)
                                * (config.n_embed // config.n_head), bias=False)
        self.out_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = self.n_embed // self.n_head

    def rotate_embeddings(self, x):
        x = x.view(*x.shape[:-1], -1, 2).flip(-1)
        x[...,0] *= -1
        return x.flatten(start_dim=-2)

    def forward(self, x):

        _, n_tokens, _ = x.shape
        head_embed = self.n_embed//self.n_head

        # Multi-Query Attention
        q, k, v = self.c_attn(x).split(
            [self.n_embed, head_embed, head_embed], dim=2)
        q = q.view((*x.shape[:2], self.n_head, -1)).permute(0, 2, 1, 3)
        k = k.view(*x.shape[:2], 1, head_embed).permute(0, 2, 1, 3)
        v = v.view(*x.shape[:2], 1, head_embed).permute(0, 2, 1, 3)

        # RoPE embeddings
        pos = 10000**((-2 * torch.arange(0, self.head_dim, 2, device=x.device) - 1)/self.head_dim)
        token_seq = torch.arange(n_tokens, dtype=pos.dtype, device=x.device).unsqueeze(1) @ pos.unsqueeze(0)
        rotary_embds = torch.cat((token_seq, token_seq), dim=-1)

        q = (q * rotary_embds.cos()) + \
            (self.rotate_embeddings(q) * rotary_embds.sin())
        k = (k * rotary_embds.cos()) + \
            (self.rotate_embeddings(k) * rotary_embds.sin())
        
        attn = F.scaled_dot_product_attention(q,k,v, dropout_p=self.dropout, is_causal=True)
        
        attn = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.resid_dropout(self.out_proj(attn))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Traditionally scale by 4, but overcompensate b/c multi-query attention
        h_dim = 4 * config.n_head * config.n_embed
        # double h_dim b/c swiglu activation
        self.fc = nn.Linear(config.n_embed, 2*h_dim, bias=False)
        self.proj = nn.Linear(h_dim, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = swiglu(self.fc(x))
        return self.dropout(self.proj(x))


class ParallelLayerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mlp = MLP(config)
        self.mlp_ln = LayerNorm(config.n_embed)

        self.multi_query_attn = MultiQueryAttention(config)
        self.mqa_ln = LayerNorm(config.n_embed)

    def forward(self, x):
        mlp_out = self.mlp(self.mlp_ln(x))
        attn_out = self.multi_query_attn(self.mqa_ln(x))
        return x + mlp_out + attn_out


class PaLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.decoder = nn.ModuleDict(dict(
            word_embds=nn.Embedding(config.vocab_size, config.n_embed),
            drop=nn.Dropout(config.dropout),
            blocks=nn.ModuleList([ParallelLayerBlock(config)
                                 for _ in range(config.n_layer)]),
            out_ln=LayerNorm(config.n_embed)
        ))

        # Set linear head weights to embedding weights according to paper
        self.ln_vocab = nn.Linear(
            config.n_embed, config.vocab_size, bias=False)
        self.ln_vocab.weight = self.decoder.word_embds.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Paper inits all weights aside from embedding and layer_norm using W ~ N(0, 1/sqrt(n_in))
        # Input embeddings get initalized to E ~ N(0,1) since layer_norm isn't applied to the embedding
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1/math.sqrt(module.in_features))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(self.decoder.word_embds.weight) # maybe make std=0.02 here

    def forward(self, x, targets=None):

        x = self.decoder.word_embds(x)
        x = self.decoder.drop(x)

        for block in self.decoder.blocks:
            x = block(x)

        x = self.decoder.out_ln(x)

        logits = self.ln_vocab(x)

        if targets is not None:
            # Paper scales pre-softmax output logits by 1/sqrt(n_embed), but I can't get this to work well
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1)

            return logits, loss
        return logits, None


@dataclass
class PaLMConfig:
    n_embed: int
    n_head: int
    dropout: float
    vocab_size: int
    n_layer: int

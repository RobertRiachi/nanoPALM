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
        self.n_embed = config.n_embed
        self.n_head = config.n_head

        head_dim = self.n_embed // self.n_head
        pos = 10000**((-2 * torch.arange(0, head_dim, 2) - 1)/head_dim)
        self.register_buffer("pos", pos)

    def rotate_embeddings(self, x):
        x = x.view(*x.shape[:-1], 2, -1)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

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
        token_seq = torch.arange(n_tokens, dtype=self.pos.dtype).unsqueeze(
            1) @ self.pos.unsqueeze(0)
        rotary_embds = torch.cat((token_seq, token_seq), dim=-1)
        q = (q * rotary_embds.cos()) + \
            (self.rotate_embeddings(q) * rotary_embds.sin())
        k = (k * rotary_embds.cos()) + \
            (self.rotate_embeddings(k) * rotary_embds.sin())

        attn_filter = (q @ k.transpose(-2, -1)) * (head_embed)**-0.5
        attn_filter = F.softmax(attn_filter, dim=-1)

        attn = self.attn_dropout(attn_filter) @ v
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
        self.device = config.device

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

    def forward(self, x):

        token_embds = self.decoder.word_embds(x)
        x = self.decoder.drop(token_embds)
        for block in self.decoder.blocks:
            x = block(x)
        x = self.decoder.out_ln(x)

        return self.ln_vocab(x)


@dataclass
class PaLMConfig:
    n_embed: int
    n_head: int
    dropout: float
    vocab_size: int
    n_layer: int
    device: str


if __name__ == "__main__":

    torch.manual_seed(1337)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    input = "This is a test"
    txt_tokens = tokenizer.encode(input)

    config = PaLMConfig(n_embed=768,
                        n_head=4,
                        dropout=0.1,
                        vocab_size=tokenizer.vocab_size,
                        n_layer=2,
                        device="cuda" if torch.cuda.is_available() else "cpu")

    palm = PaLM(config)
    out = palm(torch.tensor(txt_tokens).unsqueeze(0))

    print(out.shape)

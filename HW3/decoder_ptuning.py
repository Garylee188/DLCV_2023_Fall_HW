import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 6
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):
    # adding
    def __init__(self, cfg):
        super().__init__()
        self.img_attn = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)
        self.caption_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x_dec, x_enc):
        B_enc, N_enc, C_enc = x_enc.size() # batch, patch, embedding 
        B_dec, N_dec, C_dec = x_dec.size() # batch, context, embedding
        k_enc, v_enc  = self.img_attn(x_enc).split(self.n_embd, dim=2)
        q_dec = self.caption_attn(x_dec)
        
        # q from decoder; k,v from encoder
        k = k_enc.view(B_enc, N_enc, self.n_head, C_enc // self.n_head).transpose(1, 2)
        q = q_dec.view(B_dec, N_dec, self.n_head, C_dec // self.n_head).transpose(1, 2)
        v = v_enc.view(B_enc, N_enc, self.n_head, C_enc // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # without masked
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B_dec, N_dec, C_dec))
    
class SoftPrompt(nn.Module):
    def __init__(self, n_tokens, random_range, cfg):
        super().__init__()
        init_embed = torch.FloatTensor(n_tokens, cfg.n_embd).uniform_(-random_range, random_range)
        # self.soft_prompt = nn.Embedding(n_tokens, cfg.n_embd)
        # self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)
        self.learned_embed = nn.parameter.Parameter(init_embed)
            
    def forward(self, tokens):
        learned_embed = self.learned_embed.repeat(tokens.size(0), 1, 1)
        return torch.cat([learned_embed, tokens], 1)

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)  # adding
        self.soft_prompt = SoftPrompt(10, 0.5, cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, concat_x):
        x, x_enc = concat_x
        # Prefix tuning
        p_x = self.soft_prompt(x)
        p_x = p_x + self.attn(self.ln_1(p_x))
        p_x = p_x + self.cross_attn(self.ln_1(p_x), self.ln_3(x_enc))  # adding
        p_x = p_x + self.mlp(self.ln_2(p_x))
        concat_x = [p_x[:, 10:, :], x_enc]
        return concat_x
    
class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, x_enc):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        res_x, res_x_enc = self.transformer.h([x, x_enc])
        x = self.lm_head(self.transformer.ln_f(res_x))

        return x

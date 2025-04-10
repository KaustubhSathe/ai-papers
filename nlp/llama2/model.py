import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional



@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # number of heads for query
    n_kv_heads: Optional[int] = None # number of heads for key/value
    vocab_size: int = -1 # This will be set when loading the tokenizer
    multiple_of: int = 256 # all intermediate dims will be a multiple of this
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        assert args.vocab_size != -1, "vocab_size must be set"
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
            
            
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        self.freq_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        
    
    
    

import torch
import torch.nn as nn
from typing import Tuple


class SigLIPVisionConfig(nn.Module):
    def __init__(self, 
                 hidden_size: int = 768, 
                 intermediate_size: int = 3072, 
                 num_hidden_layers: int = 12, 
                 num_attention_heads: int = 12,
                 num_channels: int = 3,
                 image_size: int = 224,
                 patch_size: int = 16, 
                 layer_norm_eps: float = 1e-6,
                 attention_dropout: float = 0.0,
                 num_image_tokens: int = None,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        
class SigLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # this indicates no padding
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand(1, -1), persistent=False)
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embeddings(pixel_values) # [Batch_Size, Embed_Dim, Height, Width]
        embeddings = patch_embeds.flatten(2).transpose(1, 2) # [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        return embeddings
    
    
    
class SigLIPMLP(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] --> [Batch_Size, Num_Patches, Intermediate_Dim]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
    
    
class SigLIPAttention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_length, _ = hidden_states.size()
        # query states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_length, seq_length):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_length, seq_length)}, but is {attn_weights.size()}"
            )
        
        # Apply the softmax function to the attention weights row-wise
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype).to(query_states.dtype)            
        
        # Apply dropout if specified
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Multiply the attention weights with the value states
        attn_output = torch.matmul(attn_weights, value_states)
         
        if attn_output.size() != (batch_size, self.num_heads, seq_length, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(batch_size, self.num_heads, seq_length, self.head_dim)}, but is {attn_output.size()}"
            )
        
        # Reshape the output to [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        
        # Project the output to the original embedding dimension
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
        
        
        
        
        
class SigLIPEncoderLayer(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLIPAttention(config)
        self.layer_norm_1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual connection: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] --> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm_1(hidden_states)
        # self-attention: [Batch_Size, Num_Patches, Embed_Dim] --> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states)
        # residual connection: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] --> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm_2(hidden_states)
        # MLP: [Batch_Size, Num_Patches, Embed_Dim] --> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # residual connection: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        return hidden_states
                
        
class SigLIPVisionTransformer(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = SigLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] ---> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
        
    
class SigLIPVisionModel(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLIPVisionTransformer(config)
    
    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] --> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
    

        
        
        

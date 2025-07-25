import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,
      in_channels: int=3,
      patch_size: int=16,
      embedding_dim: int=768      
    ):
        super().__init__()

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride = patch_size,
            padding = 0
        )

        self.flatten = nn.Flatten(
            start_dim = 2,
            end_dim = 3,
        )

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size  == 0

        x_patched = self.patcher(x)
        x_flatten = self.flatten(x_patched)

        return x_flatten.permute(0, 2, 1)
    

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
            embedding_dim: int=768,
            num_heads: int=12,
            attention_dropout: float=0):
        super().__init__() 
        
        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True)

    def forward(self, x):
        x = self.layernorm(x)
        attention_output, _ = self.multihead_attention(query=x, key=x, value=x, need_weights =False)

        return attention_output
    
class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int=768,
                 num_heads: int=12,
                 mlp_size: int=3072,
                 mlp_dropout: float=0.1,
                 attention_dropout: float=False
                 ):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads, 
                                                     attention_dropout=attention_dropout)
        
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x
    
class ViT(nn.Module):
    def __init__(
            self,
            img_size: int=224,
            in_channels: int=3,
            patch_size: int=16,
            num_transformer_layers: int=12,
            embedding_dimension: int=768,
            mlp_size: int=3072,
            num_heads: int=12,
            attention_dropout: float=0,
            mlp_dropout: float=0.1,
            embedding_dropout: float=0.1,
            num_classes: int=1000
    ):
        super().__init__()

        assert img_size % patch_size == 0, f"image size must be dividible by patch size"

        self.num_patches =(img_size * img_size) // patch_size**2
        self.class_embeddings = nn.Parameter(data=torch.randn(1,1, embedding_dimension), requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dimension), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dimension)
        self.transformer_encoder =nn.Sequential(
            *[TransformerEncoder(embedding_dim=embedding_dimension, num_heads =num_heads, mlp_size=mlp_size, mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)]
        )
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dimension), nn.Linear(in_features=embedding_dimension, out_features=num_classes))

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embeddings.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])

        return x

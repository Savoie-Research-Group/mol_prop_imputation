import torch
import torch.nn as nn

'''
# Define the Attention Architecture with dropout
class Model(nn.Module):
    def __init__(self, pred_prop, embed_dim=128, num_properties=22, dropout_prob=0.1):
        super(Model, self).__init__()
        
        self.embedding = CustomEmbedding() 
        
    def forward(self, x):
        
        x = self.embedding(x)
        return x
'''


class Model(nn.Module):
    def __init__(self, include_quad=True, embed_dim=512, num_properties=22, dropout_prob=0.1, num_heads=16, num_feed_forward=4096, num_att_layers=4):
        super(Model, self).__init__()

        self.num_properties = num_properties
        if include_quad:
            self.embedding = CustomEmbedding(num_properties=num_properties, embed_dim=embed_dim)
        else:
            self.embedding = CustomEmbedding(num_properties=num_properties, embed_dim=embed_dim, include_quad=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=num_feed_forward,
            dropout=dropout_prob,
            batch_first=True,  # Input shape: (batch_size, num_properties, embed_dim)
            layer_norm_eps=1e-6
        )
        self.src_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_att_layers)
        self.property_decoders = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_properties)])

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, num_properties)
        Output: List of tensors of shape (batch_size, 1) for each property in pred_prop
        """
        x = self.embedding(x)  # Shape: (batch_size, num_properties, embed_dim)
        x = self.src_encoder(x)  # Shape: (batch_size, num_properties, embed_dim)
        outputs = []
        for prop_idx in range(self.num_properties):
            prop_output = self.property_decoders[prop_idx](x[:, prop_idx, :])  # Shape: (batch_size, 1)
            outputs.append(prop_output)
        output = torch.stack(outputs, dim=1).squeeze()
        return output  # List of tensors, one for each property in pred_prop


class CustomEmbedding(nn.Module):
    def __init__(self, num_properties=22, embed_dim=256, include_quad=True):
        super(CustomEmbedding, self).__init__()
        self.num_properties = num_properties
        self.embed_dim = embed_dim
        self.include_quad = include_quad
        self.property_embedding = nn.ModuleList(
            [nn.Linear(1, embed_dim) for _ in range(num_properties)]
        )
        self.token_embedding = nn.ParameterList(
            [nn.Parameter(torch.randn(embed_dim)) for _ in range(num_properties)]
        )
        self.vec_property_linear = nn.Sequential(nn.Linear(6, embed_dim), nn.Dropout(p=0))

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_properties)
        Output: Tensor of shape (batch_size, num_properties, embed_dim)
        """
        batch_size = x.size(0)
        embeddings = []

        for i in range(self.num_properties):
            property_data = x[:, i].view(-1, 1)  # Shape: (batch_size, 1)
            missing_mask = (property_data == -100).float()  # Shape: (batch_size, 1)
            linear_embedding = self.property_embedding[i](property_data)  # Shape: (batch_size, embed_dim)
            missing_embedding = self.token_embedding[i].unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, embed_dim)
            # Combine the embeddings using the mask
            combined_embedding = (1 - missing_mask) * linear_embedding + missing_mask * missing_embedding
            embeddings.append(combined_embedding)
        if self.include_quad:
            quad_encoding = self.vec_property_linear(x[:,-6:])
            embeddings.append(quad_encoding)
        output = torch.stack(embeddings, dim=1)  # Shape: (batch_size, num_properties, embed_dim)
        
        return output



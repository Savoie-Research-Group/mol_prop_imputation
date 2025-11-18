"""
Initialized from prop2prop/model_14.py
(commit: 8f23cd9ea3b630d1903fbe0a559a09093327c4a7)

Class Model --> __init__() updated
  - Remove pred_prop variable
  - Load in ChemBERTa model

Class Model --> forward() updated
  - Loop over all properties, not just pred_prop ones
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class Model(nn.Module):
    def __init__(self, include_quad=True,
                 embed_dim=512, num_properties=22, dropout_prob=0.1, num_heads=16, num_feed_forward=4096, num_att_layers=4
                 ):
        super(Model, self).__init__()

        self.num_properties = num_properties

        if include_quad:
            self.embedding = CustomEmbedding(
                num_properties=num_properties, embed_dim=embed_dim
            )
        else:
            self.embedding = CustomEmbedding(
                num_properties=num_properties, embed_dim=embed_dim, include_quad=False
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=num_feed_forward,
            dropout=dropout_prob,
            # Input shape: (batch_size, num_properties, embed_dim)
            batch_first=True,
            layer_norm_eps=1e-6
        )

        self.src_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_att_layers
        )

        self.property_decoders = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(num_properties)]
        )

        # Load ChemBERTa embedding model
        self.chemberta = AutoModel.from_pretrained(
            "DeepChem/ChemBERTa-77M-MLM"
        )
        self.chemberta_embedding = nn.Linear(
            self.chemberta.config.hidden_size, embed_dim
        )

        # Freeze ChemBERTa parameters
        for param in self.chemberta.parameters():
            param.requires_grad = False  # Prevents updates during training

    def forward(self, x, smi):
        """
        x: Input tensor of shape (batch_size, num_properties)
        smi: molecule smiles (string?)
        Output: List of tensors of shape (batch_size, 1) for each property in pred_prop
        """
        # ChemBERTa is frozen
        with torch.no_grad():
            # (batch_size, seq_len, hidden_size)
            chemberta_outputs = self.chemberta(smi)["last_hidden_state"]
            # (batch_size, hidden_size)
            cls_embedding = chemberta_outputs[:, 0, :]

        chembert_embed = self.chemberta_embedding(cls_embedding)
        chembert_embed = chembert_embed.unsqueeze(1)

        # Shape: (batch_size, num_properties, embed_dim)
        x = self.embedding(x)

        x = torch.cat((x, chembert_embed), dim=1)

        # Shape: (batch_size, num_properties, embed_dim)
        x = self.src_encoder(x)

        outputs = []
        for prop_idx in range(self.num_properties):
            # Shape: (batch_size, 1)
            prop_output = self.property_decoders[prop_idx](x[:, prop_idx, :])
            outputs.append(prop_output)

        # List of tensors, one for each property in pred_prop
        output = torch.stack(outputs, dim=1).squeeze()
        return output


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
            [nn.Parameter(torch.randn(embed_dim))
             for _ in range(num_properties)]
        )

        self.vec_property_linear = nn.Sequential(
            nn.Linear(6, embed_dim), nn.Dropout(p=0)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_properties)
        Output: Tensor of shape (batch_size, num_properties, embed_dim)
        """
        batch_size = x.size(0)
        embeddings = []

        for i in range(self.num_properties):
            # Shape: (batch_size, 1)
            property_data = x[:, i].view(-1, 1)
            # Shape: (batch_size, 1)
            missing_mask = (property_data == -100).float()

            # Shape: (batch_size, embed_dim)
            linear_embedding = self.property_embedding[i](property_data)
            # Shape: (batch_size, embed_dim)
            missing_embedding = self.token_embedding[i].unsqueeze(
                0).expand(batch_size, -1)

            # Combine the embeddings using the mask
            combined_embedding = (
                1 - missing_mask) * linear_embedding + missing_mask * missing_embedding

            embeddings.append(combined_embedding)

        if self.include_quad:
            quad_encoding = self.vec_property_linear(x[:, -6:]) # quadrupole is the last property :)
            embeddings.append(quad_encoding)

        # Shape: (batch_size, num_properties, embed_dim)
        output = torch.stack(embeddings, dim=1)

        return output

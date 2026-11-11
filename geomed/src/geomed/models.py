"""Relational geometry encoder shared by the GeoVIRA experiment runners."""
from __future__ import annotations
import torch
from torch import nn

class GeometryAttentionEncoder(nn.Module):
    """Edge-attention pooling plus a triplet branch, producing a 64-d token."""

    def __init__(self, relational_dim: int, n_edges: int, output_dim: int = 64):
        super().__init__()
        edge_dim = int(n_edges) * 4
        triplet_dim = int(relational_dim) - edge_dim
        if n_edges <= 0 or triplet_dim <= 0:
            raise ValueError("relational dimensions are incompatible with GAE")
        self.n_edges = int(n_edges)
        self.edge_dim = edge_dim
        self.triplet_dim = triplet_dim
        self.edge_projection = nn.Sequential(nn.Linear(4, 64), nn.GELU())
        self.edge_attention = nn.Linear(64, 1, bias=False)
        self.triplet_encoder = nn.Sequential(
            nn.Linear(triplet_dim, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, output_dim), nn.GELU()
        )

    def encode_branches(self, relational: torch.Tensor, edges: torch.Tensor):
        if edges.ndim != 3 or edges.shape[1:] != (self.n_edges, 4):
            raise ValueError(f"expected edges (B,{self.n_edges},4), got {tuple(edges.shape)}")
        triplets = relational[:, self.edge_dim:]
        edge_hidden = self.edge_projection(edges)
        weights = torch.softmax(self.edge_attention(edge_hidden), dim=1)
        edge_token = (weights * edge_hidden).sum(dim=1)
        angle_evidence = self.triplet_encoder(triplets)
        return edge_token, angle_evidence

    def forward(self, relational: torch.Tensor, edges: torch.Tensor,
                branch: str = "full") -> torch.Tensor:
        edge_token, triplet_token = self.encode_branches(relational, edges)
        if branch == "edge_only":
            triplet_token = torch.zeros_like(triplet_token)
        elif branch == "angle_only":
            edge_token = torch.zeros_like(edge_token)
        elif branch != "full":
            raise ValueError(f"unsupported geometry branch: {branch}")
        return self.output(torch.cat([edge_token, triplet_token], dim=1))

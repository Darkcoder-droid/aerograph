import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphConv(nn.Module):
    """
    Pure-PyTorch Dynamic Graph Convolution layer.
    Takes node features H and dynamic adjacency matrix A_t.
    H shape: (B, T, N, in_dim)
    Adj shape: (B, T, N, N) where Adj[b, t, j, i] is weight j <- i (message from i to j).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h: (B, T, N, in_dim)
        adj: (B, T, N, N)
        returns: (B, T, N, out_dim)
        """
        # Linear feature projection
        support = self.linear(h)  # (B, T, N, out_dim)
        
        # Pure PyTorch einsum: out[b, t, i, f] = sum_j adj[b, t, i, j] * support[b, t, j, f]
        # In our convention, adj[b, t, i, j] is weight for edge j -> i into node i
        out = torch.einsum("btij,btjf->btif", adj, support) + self.bias
        out = F.gelu(out)
        out = self.norm(out)
        return out

class WindSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Pollution Forecasting.
    Architecture:
      - 2 residual DynamicGraphConv layers (hidden_dim=64)
      - Reshape to (B * N, T, 64)
      - 2-layer nn.GRU (dropout=0.1)
      - MLP prediction head -> (B, N, seq_out)
    """
    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 64,
        seq_out: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.seq_out = seq_out

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Dynamic GCN Layer 1
        self.gcn1 = DynamicGraphConv(hidden_dim, hidden_dim)
        # Dynamic GCN Layer 2
        self.gcn2 = DynamicGraphConv(hidden_dim, hidden_dim)

        # Temporal GRU over time sequence T
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0
        )

        # MLP prediction head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_out)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, in_dim)
        adj: (B, T, N, N)
        returns: (B, N, seq_out) -> PM2.5 predictions for next seq_out steps
        """
        B, T, N, F_in = x.shape

        # Initial projection
        h = self.input_proj(x)  # (B, T, N, hidden_dim)

        # Residual Spatial GCN Layer 1
        h1 = self.gcn1(h, adj)
        h = h + h1

        # Residual Spatial GCN Layer 2
        h2 = self.gcn2(h, adj)
        h = h + h2  # (B, T, N, hidden_dim)

        # Prepare for Temporal GRU:
        # Transpose & reshape to treat each of (B * N) nodes as an independent temporal sequence
        # h: (B, T, N, hidden_dim) -> (B, N, T, hidden_dim) -> (B * N, T, hidden_dim)
        h_spatial = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.hidden_dim)

        # GRU forward pass
        gru_out, _ = self.gru(h_spatial)  # (B * N, T, hidden_dim)

        # Take final temporal state at t = T - 1
        last_state = gru_out[:, -1, :]  # (B * N, hidden_dim)

        # Predict future PM2.5 values: (B * N, seq_out)
        out = self.mlp_head(last_state)

        # Reshape back to (B, N, seq_out)
        out = out.view(B, N, self.seq_out)
        return out

    def forward_with_internals(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target_station_idx: int = 0,
        target_horizon_idx: int = 0,
        compute_saliency: bool = True
    ):
        """
        Runs forward inference and exposes real PyTorch activation tensors
        and input gradient saliency with respect to target station & horizon.
        
        Returns:
            pred: (B, N, seq_out)
            gcn1_act: (B, T, N, hidden_dim)
            gcn2_act: (B, T, N, hidden_dim)
            gru_hidden: (B, N, T, hidden_dim)
            saliency: (B, T, N, in_dim)
        """
        x_in = x.clone().detach().requires_grad_(True)
        B, T, N, F_in = x_in.shape

        h = self.input_proj(x_in)
        h1 = self.gcn1(h, adj)
        h = h + h1

        h2 = self.gcn2(h, adj)
        h = h + h2

        h_spatial = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.hidden_dim)
        gru_out, _ = self.gru(h_spatial)
        last_state = gru_out[:, -1, :]
        out = self.mlp_head(last_state).view(B, N, self.seq_out)

        saliency = torch.zeros_like(x_in)
        if compute_saliency and x_in.requires_grad:
            target_scalar = out[0, target_station_idx, target_horizon_idx]
            grads = torch.autograd.grad(target_scalar, x_in, retain_graph=False, create_graph=False)
            if grads and grads[0] is not None:
                saliency = grads[0].abs()

        # Extract learned weight matrix of final linear decoder layer: (seq_out, hidden_dim) = (6, 64)
        decoder_weights = self.mlp_head[-1].weight.detach()

        # Compute exact spatial message-passing contribution per upwind neighbor
        # adj is (B, T, N, N); last timestep for target station is adj[0, -1, target_station_idx, :]
        adj_target = adj[0, -1, target_station_idx, :].detach()
        neighbor_contrib = (adj_target * x_in[0, -1, :, 0].detach()).abs()

        return (
            out.detach(),
            h1.detach(),
            h2.detach(),
            gru_out.view(B, N, T, self.hidden_dim).detach(),
            saliency.detach(),
            decoder_weights,
            neighbor_contrib
        )

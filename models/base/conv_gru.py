"""
ConvGRU — temporal dynamics module for World Model.
"""
import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """
    ConvGRU cell — single step update.
    h_t = GRU(z_t, h_{t-1}) with convolutional gates.

    Gates (same as standard GRU):
      r = σ(Conv([z_t, h_{t-1}]))
      u = σ(Conv([z_t, h_{t-1}]))
      h̃ = tanh(Conv([z_t, r ⊙ h_{t-1}]))
      h_t = (1 - u) ⊙ h_{t-1} + u ⊙ h̃
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv_reset = nn.Conv2d(input_dim + hidden_dim, hidden_dim,
                                     kernel_size, padding=padding)
        self.conv_update = nn.Conv2d(input_dim + hidden_dim, hidden_dim,
                                      kernel_size, padding=padding)
        self.conv_new = nn.Conv2d(input_dim + hidden_dim, hidden_dim,
                                   kernel_size, padding=padding)

    def forward(self, z, h_prev=None):
        B = z.shape[0]
        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_dim, z.shape[2], z.shape[3],
                                 device=z.device)

        combined = torch.cat([z, h_prev], dim=1)
        r = torch.sigmoid(self.conv_reset(combined))
        u = torch.sigmoid(self.conv_update(combined))
        combined_new = torch.cat([z, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_new(combined_new))
        h_new = (1 - u) * h_prev + u * h_tilde
        return h_new


class ConvGRU(nn.Module):
    """
    ConvGRU — processes sequence in latent space.
    Input:  [B, T, D, H, W]
    Output: [B, T, D, H, W]
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvGRUCell(input_dim, hidden_dim, kernel_size)

    def forward(self, z_seq):
        B, T, D, H, W = z_seq.shape
        hidden_dim = self.cell.hidden_dim
        h = torch.zeros(B, hidden_dim, H, W, device=z_seq.device)

        h_seq = []
        for t in range(T):
            h = self.cell(z_seq[:, t], h)
            h_seq.append(h)

        return torch.stack(h_seq, dim=1)

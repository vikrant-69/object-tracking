"""
Token-Cached Temporal Memory Module for Siamese Object Tracking.

Components:
    TokenCache       — FIFO buffer holding N frames of target tokens.
    DiagonalSSMCell  — Lightweight S4-diagonal recurrent cell.
    TemporalMemoryModule — Wraps cache + SSM for training and inference.

During training, a sequence of T template token frames is fed in;
during inference, one frame at a time is pushed into the cache.
The SSM scans along the temporal dimension to produce an aggregated
target representation used as Query (Q) for the fusion layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCache:
    """
    FIFO token cache holding the last N frames of target tokens.

    Each entry has shape [num_tokens, dim].  The cache maintains a
    fixed-size circular buffer and tracks how many frames have been pushed.

    This is a plain Python object (not nn.Module) because it holds
    mutable state that should NOT be part of the model parameters
    or persistent buffers (it is reset between sequences).
    """

    def __init__(self, cache_size: int = 4, num_tokens: int = 256, dim: int = 64):
        self.cache_size = cache_size
        self.num_tokens = num_tokens
        self.dim = dim
        self.buffer = None        # [N, num_tokens, dim]  — lazily allocated
        self.write_idx = 0
        self.count = 0            # how many frames have been pushed so far

    def reset(self, device: torch.device = None, dtype: torch.dtype = torch.float32):
        """Clear the cache and allocate a fresh buffer on *device*."""
        if device is None:
            device = torch.device("cpu")
        self.buffer = torch.zeros(
            self.cache_size, self.num_tokens, self.dim,
            device=device, dtype=dtype,
        )
        self.write_idx = 0
        self.count = 0

    def push(self, tokens: torch.Tensor):
        """
        Push one frame of tokens into the cache (overwrites oldest if full).

        Args:
            tokens: [num_tokens, dim]  — single frame, unbatched.
        """
        if self.buffer is None:
            self.reset(device=tokens.device, dtype=tokens.dtype)
        self.buffer[self.write_idx] = tokens.detach() if not self.training_mode else tokens
        self.write_idx = (self.write_idx + 1) % self.cache_size
        self.count = min(self.count + 1, self.cache_size)

    @property
    def training_mode(self):
        return False  # cache always detaches to avoid temporal backprop across sequences

    def get_sequence(self) -> torch.Tensor:
        """
        Return cached tokens in chronological order.

        Returns:
            [min(count, cache_size), num_tokens, dim]
        """
        if self.count == 0:
            raise RuntimeError("TokenCache is empty; push at least one frame first.")
        n = min(self.count, self.cache_size)
        if n < self.cache_size:
            return self.buffer[:n]                                     # [n, T, D]
        # Circular buffer → reorder so oldest is first
        idx = list(range(self.write_idx, self.cache_size)) + list(range(self.write_idx))
        return self.buffer[idx]                                        # [N, T, D]

    def is_ready(self) -> bool:
        """True when at least one frame has been pushed."""
        return self.count > 0


class DiagonalSSMCell(nn.Module):
    """
    Lightweight diagonal State-Space Model cell (S4-diagonal variant).

    State equation (per token, per step):
        h[t] = A · h[t-1] + B · x[t]
        y[t] = C · h[t]   + D · x[t]

    Where A is a diagonal matrix parameterized as exp(A_log) to ensure stability
    (all eigenvalues inside the unit disk).

    Processes a temporal sequence of tokens and returns the final aggregated output.

    Args:
        dim:       token embedding dimension.
        state_dim: SSM hidden state dimension (number of diagonal entries).
    """

    def __init__(self, dim: int = 64, state_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # Learnable parameters
        # A_log: parameterize A = -exp(A_log) so eigenvalues are in (-1, 0), guaranteeing stability
        self.A_log = nn.Parameter(torch.randn(dim, state_dim) * 0.1)   # [D, S]

        # B projection: input → state space
        self.B_proj = nn.Linear(dim, dim * state_dim, bias=False)      # [D] → [D*S]

        # C projection: state space → output
        self.C_proj = nn.Linear(dim * state_dim, dim, bias=False)      # [D*S] → [D]

        # D skip connection
        self.D = nn.Parameter(torch.ones(dim))                         # [D]

        # Layer norm on output
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scan along the temporal dimension to aggregate a token sequence.

        Args:
            x: [T, num_tokens, D] — T temporal steps, each with num_tokens of dim D.
               OR [B, T, num_tokens, D] — batched variant.
        Returns:
            [num_tokens, D]  or  [B, num_tokens, D] — aggregated output from last step.
        """
        batched = x.dim() == 4
        if not batched:
            x = x.unsqueeze(0)                                         # [1, T, N, D]

        B, T, N, D = x.shape
        S = self.state_dim

        # Force float32 for SSM recurrence to prevent fp16 overflow in accumulation
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()

            # Compute stable diagonal A: values in (0, 1) for discrete recurrence
            A = torch.exp(-F.softplus(self.A_log.float()))             # [D, S], values in (0, 1)

            # Initialize hidden state
            h = torch.zeros(B, N, D, S, device=x.device, dtype=torch.float32)  # [B, N, D, S]

            # Scan through time
            for t in range(T):
                x_t = x[:, t]                                          # [B, N, D]

                # B projection
                b_t = self.B_proj(x_t).reshape(B, N, D, S)            # [B, N, D, S]

                # State update: h = A * h + B * x
                h = A.unsqueeze(0).unsqueeze(0) * h + b_t              # [B, N, D, S]

            # Read out: y = C * h + D * x_last
            h_flat = h.reshape(B, N, D * S)                            # [B, N, D*S]
            y = self.C_proj(h_flat)                                    # [B, N, D]
            y = y + self.D.float().unsqueeze(0).unsqueeze(0) * x[:, -1]  # [B, N, D] skip

            y = self.norm(y)                                           # [B, N, D]

        if not batched:
            y = y.squeeze(0)                                           # [N, D]
        return y


class TemporalMemoryModule(nn.Module):
    """
    Complete temporal memory: TokenCache + DiagonalSSMCell.

    Training mode:
        Receives a sequence [B, T, num_tokens, dim] and processes it
        through the SSM to produce aggregated Q tokens [B, num_tokens, dim].

    Inference mode:
        Pushes one frame at a time into the cache, runs SSM over the
        full cache to produce Q.

    Args:
        dim:        token embedding dimension (must match backbone output channels).
        cache_size: number of frames to hold in the FIFO cache (N).
        state_dim:  SSM hidden state dimension.
    """

    def __init__(self, dim: int = 64, cache_size: int = 4, state_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.cache_size = cache_size

        self.ssm = DiagonalSSMCell(dim=dim, state_dim=state_dim)

        # Cache is created per-sample during inference; not used during training
        self._cache = None

    def init_cache(self, num_tokens: int, device: torch.device, dtype=torch.float32):
        """Initialize a fresh cache for inference on a new sequence."""
        self._cache = TokenCache(
            cache_size=self.cache_size,
            num_tokens=num_tokens,
            dim=self.dim,
        )
        self._cache.reset(device=device, dtype=dtype)

    def push_and_aggregate(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Inference path: push one frame and return aggregated Q.

        Args:
            tokens: [num_tokens, dim]  — single unbatched frame.
        Returns:
            [num_tokens, dim] — aggregated target representation.
        """
        if self._cache is None:
            raise RuntimeError("Call init_cache() before push_and_aggregate().")
        self._cache.push(tokens)
        seq = self._cache.get_sequence()                               # [n, num_tokens, dim]
        return self.ssm(seq)                                           # [num_tokens, dim]

    def forward(self, template_sequence: torch.Tensor) -> torch.Tensor:
        """
        Training path: process a full sequence at once.

        Args:
            template_sequence: [B, T, num_tokens, dim]
                T template frames, each unfolded into tokens.
        Returns:
            [B, num_tokens, dim] — aggregated target Query tokens.
        """
        # template_sequence: [B, T, N, D]
        return self.ssm(template_sequence)                             # [B, N, D]


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    B, T, N, D = 2, 4, 256, 64

    mem = TemporalMemoryModule(dim=D, cache_size=T, state_dim=16)

    # --- Training mode ---
    seq = torch.randn(B, T, N, D)
    q = mem(seq)
    print(f"Training Q shape: {q.shape}")      # [2, 256, 64]

    # --- Inference mode ---
    mem.init_cache(num_tokens=N, device=torch.device("cpu"))
    for t in range(T):
        tokens = torch.randn(N, D)
        q_inf = mem.push_and_aggregate(tokens)
    print(f"Inference Q shape: {q_inf.shape}")  # [256, 64]

    total = sum(p.numel() for p in mem.parameters())
    print(f"Total parameters: {total:,}")

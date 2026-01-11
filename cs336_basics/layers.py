from symtable import Function

import torch
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np

class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weights, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = einx.dot('... in_features, out_features in_features -> ... out_features', x, self.weights)
        return res

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.weights, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.weights[x]

class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RmsNorm, self).__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(torch.randn(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(x**2, dim= -1, keepdim=True) / self.d_model + self.eps)
        result = x / rms * self.weights
        return result.to(in_dtype)

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.linear_1 = LinearLayer(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)
        self.linear_2 = LinearLayer(self.d_ff, self.d_model, device=self.device, dtype=self.dtype)
        self.linear_3 = LinearLayer(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)

    def get_compatible_dff(self, d_model: int) -> int:
        """
        Returns the nearest multiple of 64 to 8/3 * d_model.
        """
        raw = (8 * d_model) / 3
        rounded = int((raw + 32) // 64) * 64  # round to nearest multiple of 64
        return rounded

    def gated(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a * b 也是一样
        return torch.mul(a, b)

    # def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
    #     # torch.sigmoid(x)
    #     return 1.0 / (1.0 + torch.exp(-x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(swish(self.linear_1(x)) * self.linear_3(x))

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be divisible by 2")
        self.theta = theta
        self.dk = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # cache all sin and cos
        frequency = 1.0 / self.theta ** (2.0 * torch.arange(0, d_k / 2, device=device).float()/ d_k)
        positions = torch.arange(0, max_seq_len, device=device).float()
        frequency = einx.dot('d_k_2, max_seq_len -> max_seq_len d_k_2 ',frequency, positions)
        # print(frequency.shape)
        self.register_buffer('cos_cached', torch.cos(frequency), persistent=False)
        self.register_buffer('sin_cached', torch.sin(frequency), persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to `x`.  Works with any batch shape prefix.
        """
        if x.shape[-1] != self.dk:
            raise ValueError("RoPE only works for d_k=1")

        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        even_part = x_even * cos_pos - x_odd * sin_pos
        odd_part = x_odd * cos_pos + x_even * sin_pos

        out = torch.zeros_like(x)

        out[..., ::2] = even_part
        out[..., 1::2] = odd_part
        return out

def SoftMax(x: Tensor, dim: int = -1) -> Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_sum = torch.sum(torch.exp(x - max_x), dim=dim, keepdim=True)
    return torch.exp(x - max_x) / exp_sum

if __name__ == "__main__":
    # model = LinearLayer(10, 10, device=torch.device('cpu'))
    # print(model.state_dict())
    # x = np.ones((16, 8, 4))
    # tmp = einx.sum("a [b] c -> a c", x).shape
    # print(tmp)
    # RoPE(10000.0, 256, 10)
    x = torch.randn(2, 3, 3, device=torch.device('cpu'))
    exp_x = torch.exp(x)
    print(exp_x.shape)
    res = SoftMax(x)
    print(res)
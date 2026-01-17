from symtable import Function

import torch
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor, FloatTensor

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

def soft_max(x: Tensor, dim: int = -1) -> Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_sum = torch.sum(torch.exp(x - max_x), dim=dim, keepdim=True)
    return torch.exp(x - max_x) / exp_sum


def scaled_dot_product(
        Q: Float[Tensor, " ... seq_len_q d_k"],
        K: Float[Tensor, " ... seq_len_k d_k"],
        V: Float[Tensor, " ... seq_len_k d_v"],
        mask: Bool[Tensor, " ... seq_len_q seq_len_k"] = None) -> torch.Tensor:
    scale = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=Q.device))
    attention_score = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K) * scale
    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))
    return einx.dot("... queries keys, ... keys dim_v -> ... queries dim_v", soft_max(attention_score), V)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool, max_seq_len: int, theta: float, device = None, dtype = None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.k = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.v = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.o = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.use_rope = use_rope
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device), diagonal=0).bool()
        self.register_buffer("casual_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)
        if use_rope:
            self.RoPE = RoPE(d_k=self.d_k, max_seq_len=max_seq_len, theta=theta, device=device)

    def forward(self, x: FloatTensor, token_positions: Int[Tensor, " ... sequence_length"] = None) -> FloatTensor:
        # rearrange to multi head
        sequence_length = x.shape[-2]

        q_x = einx.rearrange("... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", self.q(x),
                             num_heads=self.num_heads)
        k_x = einx.rearrange("... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", self.k(x),
                             num_heads=self.num_heads)
        v_x = einx.rearrange("... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", self.v(x),
                             num_heads=self.num_heads)

        # apply rope to different head blocks
        if self.use_rope:
            q_x = self.RoPE(q_x, token_positions=token_positions)
            k_x = self.RoPE(k_x, token_positions=token_positions)  # auto broadcast

        scores = scaled_dot_product(q_x, k_x, v_x, self.casual_mask[..., :sequence_length, :sequence_length])
        scores = einx.rearrange("... num_heads seq_len d_k -> ... seq_len (num_heads d_k)", scores)

        return self.o(scores)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, use_rope: bool = True,
                 rope_theta: float = 10000.0,
                 device: torch.device = None, dtype: torch.dtype = None):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        self.rms_norm_1 = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.casual_multihead_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, use_rope=use_rope,
                                                             max_seq_len=max_seq_len, theta=rope_theta, device=device,
                                                             dtype=dtype)
        self.rms_norm_2 = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.ff1 = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape
        attention_score = self.casual_multihead_attention(self.rms_norm_1(x), token_positions=token_positions)
        x = x + attention_score

        ff_out = self.ff1(self.rms_norm_2(x))
        return x + ff_out

class TransformerLM(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float = 10000.0,
                 device: torch.device = None, dtype: torch.dtype = None):

        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.blocks = torch.nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                                                            max_seq_len=context_length, rope_theta=rope_theta,
                                                            use_rope=True, device=device, dtype=dtype)
                                                            for _ in range(num_layers)])
        self.norm_final = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = LinearLayer(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        if s > self.context_length:
            raise ValueError(f"seq_len {s} exceeds context_length {self.context_length}")

        x = self.token_embeddings(token_ids)

        pos = torch.arange(0, s, device=x.device)

        for block in self.blocks:
            x = block(x, token_positions = pos)

        x = self.lm_head(self.norm_final(x))
        return x

def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Copy `source` into `target` in-place, transposing `source` if that
    is what makes the shapes line up.
    """
    if source.shape == target.shape:
        target.data.copy_(source)
    elif source.T.shape == target.shape:
        target.data.copy_(source.T)
    else:
        raise ValueError(f"Shape mismatch: cannot load parameter of shape {source.shape} "
                         f"into tensor of shape {target.shape}")

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
    res = soft_max(x)
    print(res)
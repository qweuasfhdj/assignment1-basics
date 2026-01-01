import torch
import einx
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


if __name__ == "__main__":
    model = LinearLayer(10, 10, device=torch.device('cpu'))
    print(model.state_dict())
    x = np.ones((16, 8, 4))
    tmp = einx.sum("a [b] c -> a c", x).shape
    print(tmp)
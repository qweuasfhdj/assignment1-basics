import torch
import einx

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



if __name__ == "__main__":
    model = LinearLayer(10, 10, device=torch.device('cpu'))
    print(model.state_dict())
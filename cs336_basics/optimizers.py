import einx
from jaxtyping import Bool, Float, Int
from sympy.physics.units import momentum
from torch import Tensor, FloatTensor
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state["step"] += 1

                step = state.get("step")
                grad = p.grad.data
                m = state.get("m")
                v = state.get("v")
                state["m"] = beta_1 * m + (1 - beta_1) * grad

                state["v"] = beta_2 * v + (1 - beta_2) * grad * grad

                alpha = lr * ((1 - beta_2 ** step) ** 0.5) / (1 - beta_1 ** step)
                p.data -= alpha * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data -= lr * weight_decay * p.data


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=100.0)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights ** 2).mean()  # Compute a scalar loss value.
        print(f" loss: {loss.cpu().item()}",)
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
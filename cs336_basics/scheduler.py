import einx
from jaxtyping import Bool, Float, Int
from sympy.physics.units import momentum
from torch import Tensor, FloatTensor
from collections.abc import Callable, Iterable
from typing import Optional
from typing import List
import torch
import math
import numpy as np

def learning_rate_schedule(t, a_max, a_min, t_w, t_c):
    """
    :param t:
    :param a_max:
    :param a_min:
    :param t_w:
    :param t_c:
    :return:
    """
    if t < t_w:
        return t / t_w * a_max
    elif  t_w <= t <= t_c:
        return a_min + 0.5 * (1.0 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (a_max - a_min)
    else:
        return a_min

class GradientClip:
    def __init__(self, parameter : Iterable[torch.nn.Parameter], max_l2_norm, e = 1e-6):
        self.parameter = parameter
        self.max_l2_norm = max_l2_norm
        self.e = e

    def __call__(self):
        grads = [p.grad for p in self.parameter if p.grad is not None]
        all_grads = torch.cat([grad.flatten() for grad in grads])
        l2 = torch.norm(all_grads, p=2)
        if l2 > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (l2 + self.e)
            for grad in grads:
                grad.mul_(clip_coeff)

class DataLoader:
    def __init__(self, data:List[int], batch_size: int, context_length:int, shuffle=True, device="cpu"):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_len = len(data)
        self.context_length = context_length
        self.device = device

    def get_train_batch(self):
       idxs = np.random.randint(0, self.data_len - self.context_length, size=self.batch_size)
       x = np.stack([self.data[i: i + self.context_length] for i in idxs])
       y = np.stack([self.data[i + 1:  i + self.context_length + 1] for i in idxs])
       return torch.tensor(x).to(self.device), torch.tensor(y).to(self.device)

    def get_val_batch(self):
        idxs = np.arange((self.data_len - self.context_length) // self.batch_size) # 表示有多少个batch
        x = np.stack([self.data[i: i + self.context_length] for i in idxs])
        y = np.stack([self.data[i + 1:  i + self.context_length + 1] for i in idxs])
        return torch.tensor(x).to(self.device), torch.tensor(y).to(self.device)

    def __len__(self):
        return self.data_len // self.batch_size


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    checkpoint_path: str):
    checkpoint_dict = {"iteration": iteration, "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict()}
    torch.save(checkpoint_dict, checkpoint_path)

def load_checkpoint(checkpoint_path: str, model, optimizer):
    print(f"Loading checkpoint... {checkpoint_path}")
    checkpoint_dict = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    return checkpoint_dict["iteration"]


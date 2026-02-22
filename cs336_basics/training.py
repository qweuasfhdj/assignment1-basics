import einx
from jaxtyping import Bool, Float, Int
from sympy.physics.units import momentum
from torch import Tensor, FloatTensor
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import wandb
import time
import pickle as pkl
import argparse
from bpe import train_bpe
import os
from layers import TransformerLM
from optimizers import AdamW
from scheduler import save_checkpoint, load_checkpoint, DataLoader, GradientClip, learning_rate_schedule

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="mps")
parser.add_argument("--file_name", type=str, default="owt")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--train_steps", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--no-rmsnorm', dest='use_rmsnorm', action='store_false',
                    help="Disable RMSNorm and use LayerNorm instead")
parser.set_defaults(use_rmsnorm=True)
args = parser.parse_args()

device = args.device
epochs = args.epochs
train_steps = args.train_steps
batch_size = args.batch_size
file_name = args.file_name

timestamp = time.strftime("%Y%m%d_%H%M%S")
wandb.login()
wandb.login()
run = wandb.init(project="cs336_final_train",
                 config={
                     # Experiment
                     "experiment_name": f"tinystories_17M_{timestamp}",
                     "total_tokens_processed": 327_680_000,

                     # Data
                     "train_data_path": f"../data/{file_name}_train.txt",
                     "valid_data_path": f"../data/{file_name}_valid.txt",
                     "vocab_path": "vocab.json",
                     "merges_path": "merges.txt",

                     # Model
                     "vocab_size": 10000,
                     "context_length": 256,
                     "d_model": 512,
                     "d_ff": 1344,
                     "n_layers": 4,
                     "n_heads": 16,
                     "rope_theta": 10000.0,

                     # Training
                     "batch_size": batch_size,  # Adjust based on your GPU memory
                     # "learning_rate": 3e-5,
                     # 学习率退火相关参数
                     "initial_lr": 3e-5,
                     "max_learning_rate": 3e-5,
                     "min_learning_rate": 1e-5,
                     "lr_warmup_steps": 2000,
                     "cosine_cycle_iters": 10000,

                     # 优化器相关参数
                     "weight_decay": 0.1,
                     "adam_beta1": 0.9,
                     "adam_beta2": 0.95,
                     "eps": 1e-8,

                     # 梯度裁剪
                     "grad_clip": 1.0,

                     # 训练相关参数
                     "epochs": epochs,
                     "train_steps": train_steps,

                     # Logging & Checkpointing
                     "log_interval": 20,
                     "val_interval": 20,
                     "checkpoint_interval": 60,
                     "checkpoint_dir": "checkpoints",
                 }
                 )
config = run.config

# 训练BPE分词器
special_tokens = ["<|endoftext|>"]
data_path = config["train_data_path"]
vocab_size = config["vocab_size"]
vocab, merges = train_bpe(data_path, vocab_size, special_tokens)
print("已经训练好BPE分词器")

# 从 vocab.pkl 加载词汇表
with open("vocab.pkl", "rb") as f:
    # pickle.load 会自动恢复字典，并且值是 bytes 类型
    vocab = pkl.load(f)


device = torch.device(device if torch.mps.is_available() else "cpu")
data_path = config["train_data_path"]
vocab_size = config["vocab_size"]


if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 创建成功")
else:
    print(f"文件夹 '{folder_name}' 已存在")


#直接导入编码后的数据
with open("encoded_ids_train.pkl", "rb") as f:
    train_encode_ids = pkl.load(f)
with open("encoded_ids_valid.pkl", "rb") as f:
    valid_encode_ids = pkl.load(f)
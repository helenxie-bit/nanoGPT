#!/bin/bash

# Basic configuration
echo "Running basic configuration (n_embd/n_head = 384/6 = 64, CausalSelfAttention, standard MLP)..."
python train.py config/train_shakespeare_char.py --wandb_log=True

# Configuration with different dimensionality of q, k, v
echo "Running configuration (n_embd/n_head = 192/6 = 32)..."
python train.py config/train_shakespeare_char.py --n_embd=192 --n_head=6 --wandb_log=True

echo "Running configuration (n_embd/n_head = 48/6 = 8)..."
python train.py config/train_shakespeare_char.py --n_embd=48 --n_head=6 --wandb_log=True

# Configuration with sliding window attention
echo "Running configuration with sliding window attention (window_size = 100)..."
python train.py config/train_shakespeare_char.py --sliding_window_attention=True --window_size=100 --wandb_log=True

echo "Running configuration with sliding window attention (window_size = 10)..."
python train.py config/train_shakespeare_char.py --sliding_window_attention=True --window_size=10 --wandb_log=True

echo "Running configuration with sliding window attention (window_size = 3)..."
python train.py config/train_shakespeare_char.py --sliding_window_attention=True --window_size=3 --wandb_log=True

# Configuration with different MLP
echo "Running configuration with different MLP..."
python train.py config/train_shakespeare_char.py --mlp_type=type2 --wandb_log=True

# Configuration with register tokens
echo "Running configuration with register tokens (n_regist = 1)..."
python train.py config/train_shakespeare_char.py --n_regist=1 --wandb_log=True

echo "Running configuration with register tokens (n_regist = 5)..."
python train.py config/train_shakespeare_char.py --n_regist=5 --wandb_log=True

# Configuration with sliding window attention and register tokens
echo "Running configuration with sliding window attention and register tokens (window_size = 3, n_regist=1)..."
python train.py config/train_shakespeare_char.py --sliding_window_attention=True --window_size=4 --n_regist=1 --wandb_log=True

# Configuration with different softmax layer
echo "Running configuration with different softmax layer..."
python train.py config/train_shakespeare_char.py --custom_softmax=True --wandb_log=True
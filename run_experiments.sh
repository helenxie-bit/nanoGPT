#!/bin/bash

# Install necessary packages
# echo "Installing necessary packages..."
# pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib

# Prepare data
# echo "Preparing data..."
# python data/shakespeare_char/prepare.py

# Basic configuration
# echo "Running basic configuration (q, k size = 64, CausalSelfAttention, standard MLP)..."
# python train.py config/train_shakespeare_char.py --wandb_log=True --wandb_run_name="run0"

# Configuration with different dimensionality of q, k
echo "Running configuration (q, k size = 32)..."
python train.py config/train_shakespeare_char.py --head_size=32  --wandb_log=True --wandb_run_name="run1(q,k_size_32)"

echo "Running configuration (q, k size = 8)..."
python train.py config/train_shakespeare_char.py --head_size=8 --wandb_log=True --wandb_run_name="run2(q,k_size_8)"

# Configuration with sliding window attention
echo "Running configuration with sliding window attention (window_size = 100)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=100 --wandb_log=True --wandb_run_name="run3(window_size_100)"

echo "Running configuration with sliding window attention (window_size = 10)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=10 --wandb_log=True --wandb_run_name="run4(window_size_10)"

echo "Running configuration with sliding window attention (window_size = 3)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=3 --wandb_log=True --wandb_run_name="run5(window_size_3)"

echo "Running configuration with sliding window attention (window_size = 1)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=1 --wandb_log=True --wandb_run_name="run6(window_size_1)"

echo "Running configuration with sliding window attention (window_size = 0)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=0 --wandb_log=True --wandb_run_name="run7(window_size_0)"

# Configuration with different MLP
# echo "Running configuration with different MLP..."
# python train.py config/train_shakespeare_char.py --mlp_type="mlp2" --wandb_log=True --wandb_run_name="run8(mlp2)"

# Configuration with register tokens
echo "Running configuration with register tokens (n_regist = 1)..."
python train.py config/train_shakespeare_char.py --n_regist=1 --wandb_log=True --wandb_run_name="run9(n_regist_1)"

echo "Running configuration with register tokens (n_regist = 5)..."
python train.py config/train_shakespeare_char.py --n_regist=5 --wandb_log=True --wandb_run_name="run10(n_regist_5)"

echo "Running configuration with register tokens (n_regist = 10)..."
python train.py config/train_shakespeare_char.py --n_regist=10 --wandb_log=True --wandb_run_name="run11(n_regist_10)"

# Configuration with sliding window attention and register tokens
echo "Running configuration with sliding window attention and register tokens (window_size = 3, n_regist=1)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=3 --n_regist=1 --wandb_log=True --wandb_run_name="run12(window_size_3_n_regist_1)"

echo "Running configuration with sliding window attention and register tokens (window_size = 3, n_regist=5)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=3 --n_regist=5 --wandb_log=True --wandb_run_name="run13(window_size_3_n_regist_5)"

echo "Running configuration with sliding window attention and register tokens (window_size = 3, n_regist=10)..."
python train.py config/train_shakespeare_char.py --is_causal=False --window_size=3 --n_regist=10 --wandb_log=True --wandb_run_name="run14(window_size_3_n_regist_10)"

# Configuration with different softmax layer
# echo "Running configuration with different softmax layer..."
# python train.py config/train_shakespeare_char.py --abs_softmax=True --wandb_log=True --wandb_run_name="run15(abs_softmax)"
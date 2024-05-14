import matplotlib.pyplot as plt
import re
import argparse

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    train_loss = re.findall(r'step (\d+): train loss ([\d\.]+)', content)
    val_loss = re.findall(r'step (\d+): val loss ([\d\.]+)', content)
    return train_loss, val_loss

def plot_loss(train_losses, val_losses, title):
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        train_iters, train_loss_values = zip(*train_loss)
        val_iters, val_loss_values = zip(*val_loss)
        plt.plot(train_iters, train_loss_values, label=f'Run {i+1} Train Loss')
        plt.plot(val_iters, val_loss_values, label=f'Run {i+1} Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--file_paths', nargs='+', type=str)
args = parser.parse_args()
train_losses = []
val_losses = []
for file_path in args.file_paths:
    train_loss, val_loss = parse_log_file(file_path)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

plot_loss(train_losses, val_losses, 'Train and Val Loss')
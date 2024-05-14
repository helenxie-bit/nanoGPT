import matplotlib.pyplot as plt
import re
import argparse

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    train_loss = [(int(step), float(loss)) for step, loss in re.findall(r'step (\d+): train loss ([\d\.]+)', content)]
    val_loss = [(int(step), float(loss)) for step, loss in re.findall(r'step (\d+): train loss [\d\.]+, val loss ([\d\.]+)', content)]
    return train_loss, val_loss

def plot_loss(train_losses, val_losses, title):
    # Plot train loss
    plt.figure()  # create a new figure for train loss
    for i, train_loss in enumerate(train_losses):
        train_iters, train_loss_values = zip(*train_loss)
        plt.plot(train_iters, train_loss_values, label=f'Run {i+1} Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title + ' - Train Loss')
    plt.legend()
    plt.savefig('train_loss_plot.png')
    plt.show()

    # Plot val loss
    plt.figure()  # create a new figure for val loss
    for i, val_loss in enumerate(val_losses):
        if val_loss:  # check if val_loss is not empty
            val_iters, val_loss_values = zip(*val_loss)
            plt.plot(val_iters, val_loss_values, label=f'Run {i+1} Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title + ' - Val Loss')
    plt.legend()
    plt.savefig('val_loss_plot.png')
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
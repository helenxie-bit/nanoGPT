import matplotlib.pyplot as plt
import re
import argparse

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    train_loss = [(int(step), float(loss)) for step, loss in re.findall(r'step (\d+): train loss ([\d\.]+)', content)]
    val_loss = [(int(step), float(loss)) for step, loss in re.findall(r'step (\d+): train loss [\d\.]+, val loss ([\d\.]+)', content)]
    return train_loss, val_loss


# ----------------------------
# Plot the results of different dimensionality
file_paths = ["output.log", "output_head_size_32.log", "output_head_size_8.log"]
train_losses = []
val_losses = []
for file_path in file_paths:
    train_loss, val_loss = parse_log_file(file_path)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot train loss
plt.figure()  # create a new figure for train loss
for train_loss, configuration in zip(train_losses, ['size of q,k = 64', 'size of q,k = 32', 'size of q,k = 8']):
    train_iters, train_loss_values = zip(*train_loss)
    plt.plot(train_iters, train_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('train_loss_plot_1.png')
plt.show()

# Plot val loss
plt.figure()  # create a new figure for val loss
for val_loss, configuration in zip(val_losses, ['size of q,k = 64', 'size of q,k = 32', 'size of q,k = 8']):
    val_iters, val_loss_values = zip(*val_loss)
    plt.plot(val_iters, val_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Val Loss')
plt.legend()
plt.savefig('val_loss_plot_1.png')
plt.show()


# ----------------------------
# Plot the results of sliding window attention
file_paths = ["output.log", "output_window_size_100.log", "output_window_size_10.log", "output_window_size_3.log"]
train_losses = []
val_losses = []
for file_path in file_paths:
    train_loss, val_loss = parse_log_file(file_path)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot train loss
plt.figure()  # create a new figure for train loss
for train_loss, configuration in zip(train_losses, ['causal attention', 'sliding window attention (window_size=100)', 'sliding window attention (window_size=10)', 'sliding window attention (window_size=3)']):
    train_iters, train_loss_values = zip(*train_loss)
    plt.plot(train_iters, train_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('train_loss_plot_2.png')
plt.show()

# Plot val loss
plt.figure()  # create a new figure for val loss
for val_loss, configuration in zip(val_losses, ['causal attention', 'sliding window attention (window_size=100)', 'sliding window attention (window_size=10)', 'sliding window attention (window_size=3)']):
    val_iters, val_loss_values = zip(*val_loss)
    plt.plot(val_iters, val_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Val Loss')
plt.legend()
plt.savefig('val_loss_plot_2.png')
plt.show()


# ----------------------------
# Plot the results of different mlp
file_paths = ["output.log", "output_mlp2.log"]
train_losses = []
val_losses = []
for file_path in file_paths:
    train_loss, val_loss = parse_log_file(file_path)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot train loss
plt.figure()  # create a new figure for train loss
for train_loss, configuration in zip(train_losses, ['mlp1', 'mlp2']):
    train_iters, train_loss_values = zip(*train_loss)
    plt.plot(train_iters, train_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('train_loss_plot_3.png')
plt.show()

# Plot val loss
plt.figure()  # create a new figure for val loss
for val_loss, configuration in zip(val_losses, ['mlp1', 'mlp2']):
    val_iters, val_loss_values = zip(*val_loss)
    plt.plot(val_iters, val_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Val Loss')
plt.legend()
plt.savefig('val_loss_plot_3.png')
plt.show()

'''
# ----------------------------
# Plot the results of adding register tokens
file_paths = ["output_1.log", "output_8.log", "output_9.log"]
train_losses = []
val_losses = []
for file_path in file_paths:
    train_loss, val_loss = parse_log_file(file_path)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot train loss
plt.figure()  # create a new figure for train loss
for train_loss, configuration in zip(train_losses, ['n_regist=0', 'n_regist=1', 'n_regist=5']):
    train_iters, train_loss_values = zip(*train_loss)
    plt.plot(train_iters, train_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('train_loss_plot_5.png')
plt.show()

# Plot val loss
plt.figure()  # create a new figure for val loss
for val_loss, configuration in zip(val_losses, ['n_regist=0', 'n_regist=1', 'n_regist=5']):
    val_iters, val_loss_values = zip(*val_loss)
    plt.plot(val_iters, val_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Val Loss')
plt.legend()
plt.savefig('val_loss_plot_5.png')
plt.show()


# ----------------------------
# Plot the results of adding register tokens + sliding window attention
file_paths = ["output_6.log", "output_8.log", "output_10.log"]
train_losses = []
val_losses = []
for file_path in file_paths:
    train_loss, val_loss = parse_log_file(file_path)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot train loss
plt.figure()  # create a new figure for train loss
for train_loss, configuration in zip(train_losses, ['sliding window attention (window_size=3)', 'n_regist=1', 'sliding window attention (window_size=3) & n_regist=1']):
    train_iters, train_loss_values = zip(*train_loss)
    plt.plot(train_iters, train_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('train_loss_plot_6.png')
plt.show()

# Plot val loss
plt.figure()  # create a new figure for val loss
for val_loss, configuration in zip(val_losses, ['sliding window attention (window_size=3)', 'n_regist=1', 'sliding window attention (window_size=3) & n_regist=1']):
    val_iters, val_loss_values = zip(*val_loss)
    plt.plot(val_iters, val_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Val Loss')
plt.legend()
plt.savefig('val_loss_plot_6.png')
plt.show()


# ----------------------------
# Plot the results of different softmax
file_paths = ["output_1.log", "output_11.log"]
train_losses = []
val_losses = []
for file_path in file_paths:
    train_loss, val_loss = parse_log_file(file_path)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot train loss
plt.figure()  # create a new figure for train loss
for train_loss, configuration in zip(train_losses, ['standard softmax', 'changed softmax']):
    train_iters, train_loss_values = zip(*train_loss)
    plt.plot(train_iters, train_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('train_loss_plot_7.png')
plt.show()

# Plot val loss
plt.figure()  # create a new figure for val loss
for val_loss, configuration in zip(val_losses, ['standard softmax', 'changed softmax']):
    val_iters, val_loss_values = zip(*val_loss)
    plt.plot(val_iters, val_loss_values, label=f'{configuration}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Val Loss')
plt.legend()
plt.savefig('val_loss_plot_7.png')
plt.show()
'''
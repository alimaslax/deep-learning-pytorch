import matplotlib.pyplot as plt
import wandb

# Initialize W&B in offline mode
wandb.init(mode="offline")

# Read data from file
with open('../../train/lora/training-run.txt', 'r') as f:
    data = f.readlines()

# Extract the data
train_loss = []
val_loss = []
for line in data:
    if 'Train loss' in line:
        train_loss.append(float(line.split('Train loss')[1].split(',')[0].strip()))
    elif 'Val loss' in line:
        val_loss.append(float(line.split('Val loss')[1].split(',')[0].strip()))

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the train loss data
def plot_train_loss():
    ax.plot(range(len(train_loss)), train_loss, label='Train Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Train Loss Over Iterations')
    ax.legend()

# Plot the val loss data
def plot_val_loss():
    ax.plot(range(len(val_loss)), val_loss, label='Val Loss')
    ax.set_xlabel('Iteration (x10)')
    ax.set_ylabel('Loss')
    ax.set_title('Val Loss Over Iterations')
    ax.legend()

# Call the functions to plot the data
plot_train_loss()
plot_val_loss()

# Log train and val loss to W&B (will be stored locally in offline mode)
wandb.log({'Train Loss': train_loss, 'Val Loss': val_loss})

# Show the plot
plt.show()
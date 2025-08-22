import matplotlib.pyplot as plt  # pip install -U matplotlib
import glob

def makeLinearGraph(train_loss, val_loss):
    loops = len(train_loss)
    # Determine step based on the num training loops
    steps = loops // 17
    
    # Create a list of indices and ensure the final index is included
    epochs = list(range(1, loops + 1, steps))
    # if epochs[-1] != len(train_loss):
    #     epochs.append(len(train_loss))

    # Create a new figure
    plt.figure(figsize=(8, 5), dpi=100)

    # Plot training and validation loss
    plt.plot(epochs, [train_loss[i-1] for i in epochs], 'b--', label='Training Loss')
    plt.plot(epochs, [val_loss[i-1] for i in epochs], 'r-', label='Validation Loss')

    # Graph title and labels
    plt.title('Training vs. Validation Loss', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    # Generate a unique filename
    counter = 1
    while glob.glob(f"model{counter}*loopsPlot.png"):
        counter += 1
    fileName = f"model{counter}_{loops}loopsPlot.png"

    # Save the updated figure
    plt.savefig(fileName)
    print(f"Graph saved as {fileName}")
    
    return fileName

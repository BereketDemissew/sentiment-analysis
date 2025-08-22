import matplotlib.pyplot as plt  # pip install -U matplotlib
import glob

def makeLinearGraph(train_loss, val_loss, accuracy):
    loops = len(train_loss)
    # Determine step based on the num training loops
    if loops > 400:
        steps = 50
    elif loops < 10:
        steps = 1
    elif loops > 200:
        steps = 20
    # elif loops > 10:
    #     step = 10
    else:
        steps = 10
    
    # Create a list of indices and ensure the final index is included
    epochs = list(range(1, loops + 1, steps))
    # if epochs[-1] != len(train_loss):
    #     epochs.append(len(train_loss))

    # Create a new figure
    plt.figure(figsize=(8, 5), dpi=100)

    # Plot training and validation loss
    plt.plot(epochs, [train_loss[i-1] for i in epochs], 'b--', label='Training Loss')
    plt.plot(epochs, [val_loss[i-1] for i in epochs], 'r--', label='Validation Loss')
    plt.plot(epochs, [accuracy[i-1] for i in epochs], 'm--', label='Accuracy')

    # Graph title and labels
    plt.title('Training Loss, Validation Loss, Accuracy', fontsize=16)
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

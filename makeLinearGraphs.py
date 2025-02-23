import matplotlib.pyplot as plt # pip install -U matplotlib
import glob
import numpy as np # pip install numpy

def makeLinearGraph(train_loss, val_loss, loops):
    epochs = range(1, len(train_loss) + 1)  # Create a range for x-axis

    plt.figure(figsize=(8, 5), dpi=100)  # Set figure size

    # Plot training and validation loss
    plt.plot(epochs, train_loss, 'b--', label='Training Loss')  # Dashed blue line
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')   # Solid red line

    # Graph title and labels
    plt.title('Training vs. Validation Loss', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set x-axis ticks
    plt.xticks(epochs)

    # Add legend and grid
    plt.legend()
    plt.grid(True)

    # Show the plot
    # plt.show()

    counter = 1
    fileName = ""
    while glob.glob(f"model{counter}*loopsPlot.png"): # uses glob to use astrix to allow for overwriting files to only check for model # and not epoch -- WC
        counter+=1
    fileName = f"model{counter}_{loops}loopsPlot.png"
    plt.savefig(fileName)
    print(f"dipstick graph saved as {fileName}")
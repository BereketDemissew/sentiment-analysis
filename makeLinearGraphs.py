import matplotlib.pyplot as plt  # pip install -U matplotlib
import glob
import numpy as np  # pip install numpy

def makeLinearGraph(train_loss, val_loss, loops):
    # Create a new figure
    plt.figure(figsize=(8, 5), dpi=100)
    epochs = range(1, loops + 1)  # Create a range for x-axis

    # Plot training and validation loss
    plt.plot(epochs, train_loss, 'b--', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')

    # Graph title and labels
    plt.title('Training vs. Validation Loss', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set x-axis ticks
    if loops <= 10:
        plt.xticks(np.arange(0, loops, step=1))
    elif loops <= 100:
        plt.xticks(np.arange(0, loops, step=10))
    elif loops > 400 and loops < 800:
        plt.xticks(np.arange(0, loops, step=50))
    else:
        plt.xticks(np.arange(0, loops, step=100))

    # Add legend and grid
    plt.legend()
    plt.grid(True)

    # Display the plot non-blocking for 5 seconds
    plt.show(block=False)
    plt.pause(5)

    # Get the current figure and force a canvas update
    fig = plt.gcf()
    fig.canvas.draw()

    # Generate a unique filename
    counter = 1
    while glob.glob(f"model{counter}*loopsPlot.png"):
        counter += 1
    fileName = f"model{counter}_{loops}loopsPlot.png"

    # Save the updated figure
    fig.savefig(fileName)
    print(f"Graph saved as {fileName}")

    plt.close(fig)
    return fileName



# import matplotlib.pyplot as plt # pip install -U matplotlib
# import glob
# import numpy as np # pip install numpy

# def makeLinearGraph(train_loss, val_loss, loops):
#     # epochs = len(train_loss)  # Create a range for x-axis
#     epochs = list(range(1, loops + 1)) # Create a range for x-axis

#     plt.figure(figsize=(8, 5), dpi=100)  # Set figure size

#     # Plot training and validation loss
#     plt.plot(epochs, train_loss, 'b--', label='Training Loss')  # Dashed blue line
#     plt.plot(epochs, val_loss, 'r-', label='Validation Loss')   # Solid red line

#     # Graph title and labels
#     plt.title('Training vs. Validation Loss', fontsize=16)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')

#     # Set x-axis ticks
#     if loops <= 100:
#       plt.xticks(np.arange(0, loops, step=10))
#     elif loops > 400 and loops < 800:
#       plt.xticks(np.arange(0, loops, step=50))
#     else:
#       plt.xticks(np.arange(0, loops, step=100))

#     # Add legend and grid
#     plt.legend()
#     plt.grid(True)

#     # Show the plot
#     plt.show(block=False)
#     plt.pause(5)


#     counter = 1
#     fileName = ""
#     while glob.glob(f"model{counter}*loopsPlot.png"): # uses glob to use astrix to allow for overwriting files to only check for model # and not epoch -- WC
#         counter+=1
#     fileName = f"model{counter}_{loops}loopsPlot.png"
#     plt.savefig(fileName)
#     print(f"dipstick graph saved as {fileName}")

#     plt.close()
#     return fileName

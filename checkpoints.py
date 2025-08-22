import glob
import torch

def saveCheckpoint(checkpoint, loops):
    print('saving checkpoint...')

    counter = 1
    fileName = ""
    while glob.glob(f"model{counter}*.pth.tar"): # uses glob to use astrix to allow for overwriting files to only check for model # and not epoch -- WC
        counter+=1
    fileName = f"model{counter}_{loops}loops.pth.tar"
    torch.save(checkpoint, fileName)
    print(f"dipstick model saved as {fileName}")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def loadCheckpoint(model, optimizer, device):
    while True:
        try:
            modelName = input("File name: ")
            torch.serialization.add_safe_globals([modelName])
            checkpoint = torch.load(modelName, map_location = device, weights_only=False) # map_location is where the storage should be remapped to
            break
        except FileNotFoundError:
            print(f"{modelName} is not a valid file, please try again")
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    trainingLosses = checkpoint.get('trainingLosses', []) # saves an empty list if losses is missing
    validationLosses = checkpoint.get('validationLosses', [])
    accuracyList = checkpoint.get('accuracyList', [])
    numTrainingLoops = checkpoint.get('loops', 0)
    return model, optimizer, trainingLosses, validationLosses, numTrainingLoops, accuracyList

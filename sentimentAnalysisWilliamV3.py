import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

# File imports
import trainModel
import lexigraphicOrder
import checkpoints
import makeLists
import makeLinearGraphs
import calculateAccuracy
import sendEmail
import manualTestingReviews

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Reviewer(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.fc1 = nn.Linear(dim, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        embed = self.embedding(x)          # (batch, seq_len, dim)
        mean = torch.mean(embed, axis=1)   # (batch, dim)
        out = self.fc1(mean)               # (batch, 16)
        out = self.relu(out)
        out = self.fc2(out)                # (batch, 1) logits
        return out   # raw logits

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# make lists code
print('starting')
start_lister = time.time()
review, score, test_review, test_score, val_review, val_score = makeLists.make_lists()
end_lister = time.time()
print(f"takes {round(end_lister - start_lister, 2)} seconds to make lists")

print('lexigraphic order')
# lexigraphic order code -- turns words into numbers for NN
start_lex = time.time()
vocab, padded, hashing, score = lexigraphicOrder.lex_order(review, score)
padd_width = len(padded[0])
test_review = lexigraphicOrder.lex_order_new(hashing, test_review, padd_width) # tokenize the test review
val_review = lexigraphicOrder.lex_order_new(hashing, val_review, padd_width)
end_lex = time.time()
print(f"takes {round(end_lex - start_lex, 2)} seconds to make the lex order")

print('creating model')
# moved some of training loop function out to allow loadCheckpoint to work
model = Reviewer(vocab, 256)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters()) # automatic gradient descent
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # 'min' to decrease metric, factor - LR multiplicative scale when plateau, patience - wait 3 epochs before changing LR by factor
trainingLosses = []
validationLosses = []
accuracyList = []
previousTrainingLoops = 0
manuallyTestReviews = False

print('training loop')
# loads model and optimizer from checkpoint
loadCheckpoint = input("Would you like to load a checkpoint? (y/n): ")
if (loadCheckpoint == 'y'):
    [model, optimizer, trainingLosses, validationLosses, previousTrainingLoops, accuracyList] = checkpoints.loadCheckpoint(model, optimizer, device)
checkpoint = trainModel.training_loop(model, optimizer, scheduler, padded, score, val_review, val_score, trainingLosses, validationLosses, previousTrainingLoops, test_review, test_score, accuracyList, device)

loops = checkpoint['loops']
trainingLosses = checkpoint['trainingLosses']
validationLosses = checkpoint['validationLosses']
accuracyList = checkpoint['accuracyList']
batchSize = checkpoint['batchSize']
print("total loops:", loops)
print('trainingLosses length:', len(trainingLosses))
print('validationLosses length:', len(validationLosses))
print ('accuracyList length:', len(accuracyList))

checkpoint['model'] = model.state_dict() # doesn't change in training_loop so just adding to checkpoint manually
checkpoint['optimizer'] = optimizer.state_dict() # doesn't change in training_loop so just adding to checkpoint manually
if loops != previousTrainingLoops: # only runs if training loops > 0
    accuracy = calculateAccuracy.compute_accuracy(model, test_review, test_score, device, batchSize) # test dataset/accuracy
    pngFileName = makeLinearGraphs.makeLinearGraph(trainingLosses, validationLosses, accuracyList) # make plot of training and validation from loops
    try:
        sendEmail.sendEmail(loops, round(trainingLosses[-1], 2), round(validationLosses[-1], 2), True, pngFileName, accuracy) # boolean is for trainingCompleted
    except:
        print("unable to connect to server")
    checkpoints.saveCheckpoint(checkpoint, loops)
if manuallyTestReviews:
    manualTestingReviews.testing_input(hashing, model, padd_width) # Broken, need to fix
# print('goodbye sweet world...')
# os.system('shutdown -s')
print("done")

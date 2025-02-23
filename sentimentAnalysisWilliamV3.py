import csv
import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import matplotlib.pyplot as plt
import re
import glob

# File imports
import trainModel
import lexigraphicOrder
import checkpoints
import makeLists
import makeLinearGraphs
import calculateAccuracy

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class reviewer(nn.Module): # new porblem found: high af loss 1/10
  def __init__(self, vocab, dim):
    super().__init__()
    self.emdding = nn.Embedding(vocab, dim) # caused error with space used 1/6 - fixed 1/7 (error is from training not module)
    self.lin = nn.Linear(dim, 1)
    self.lin2 = nn.Linear(16, 1)
    self.sig = nn.Sigmoid()
  def forward(self, x):
    embed = self.emdding(x)
    mean = torch.mean(embed, axis = 1)
    sigs = self.sig(mean)
    lin = self.lin(sigs)
    sig = self.sig(lin)
    return sig
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# make lists code
start_lister = time.time()
review, score, test_review, test_score, val_review, val_score = makeLists.make_lists()
end_lister = time.time()
print(f"takes {round(end_lister - start_lister, 2)} seconds to make lists")

# lexigraphic order code -- turns words into numbers for NN
start_lex = time.time()
vocab, padded, hashing, score = lexigraphicOrder.lex_order(review, score)
padd_width = len(padded[0])
test_review = lexigraphicOrder.lex_order_new(hashing, test_review, padd_width)# tokenize the test review
val_review = lexigraphicOrder.lex_order_new(hashing, val_review, padd_width)
end_lex = time.time()
print(f"takes {round(end_lex - start_lex, 2)} seconds to make the lex order")

# moved some of training loop function out to allow loadCheckpoint to work
model = reviewer(vocab, 256)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
trainingLosses = []
validationLosses = []
previousTrainingLoops = 0


# loads model and optimizer from checkpoint
loadCheckpoint = input("Would you like to load a checkpoint? (y/n): ")
if (loadCheckpoint == 'y'):
    [model, optimizer, trainingLosses, validationLosses, previousTrainingLoops] = checkpoints.loadCheckpoint(model, optimizer, device)
start_model = time.time()
checkpoint = trainModel.training_loop(model, optimizer, vocab, padded, score, val_review, val_score, test_review, test_score, trainingLosses, validationLosses, previousTrainingLoops, device)
end_model = time.time()
print(f"takes {end_model - start_model} seconds for training")

loops = checkpoint['loops']
trainingLosses = checkpoint['trainingLosses']
validationLosses = checkpoint['validationLosses']

checkpoint['model'] = model.state_dict()
checkpoint['optimizer'] = optimizer.state_dict()
# model = checkpoint['model']
# optimizer = checkpoint['optimizer']
# loops = checkpoint['loops']
# trainingLosses = checkpoint['trainingLosses']
# validationLosses = checkpoint['validationLosses']

# model.load_state_dict(checkpoint['model'])
# model = model.to(device)
# optimizer.load_state_dict(checkpoint['optimizer'])
# trainingLosses = checkpoint.get('trainingLosses', []) # saves an empty list if losses is missing
# validationLosses = checkpoint.get('validationLosses', [])
# numTrainingLoops = checkpoint.get('loops', 0)

makeLinearGraphs.makeLinearGraph(trainingLosses, validationLosses, loops)
# test dataset/accuracy
calculateAccuracy.compute_accuracy(model, test_review, test_score, loops, device)

checkpoints.saveCheckpoint(checkpoint, loops + previousTrainingLoops)


# print("Do you want to test the data with input data or some training data?")
# answer = input()
# if answer in {"input"}:
#     testing_input(review, my_model)
print("done")

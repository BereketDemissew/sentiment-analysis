import torch # pip install torch torchvision torchaudio
import time
import numpy as np

# Purpose: processes a list of text reviews, creating a numerical representation of the words in the reviews, needed for ML
# Parameters: review (list of strings)
# Return values: vocab, padded, hashing
def lex_order(reviews, score):
    words = set()
    for i in reviews: # problem: crashed because I traid to make another list of lists of it so I can maintain the order but that was too much for memory 1/3
        for n in i.split(): # solution found: just use the input reviews as the order and don't make a list of lists 1/7
            words.add(n)
    words = sorted(list(words))
    vocab = len(words) + 1
    hashing = {}
    order = 1
    for i in words:
        hashing[i] = order
        order += 1
    lex = []

   # make padding for training
    for review in reviews:
        lister = []
        for words in review.split():

            lister.append(hashing[words])
        lex.append(torch.tensor(lister, dtype=torch.long))
    padded = torch.nn.utils.rnn.pad_sequence(lex, batch_first=True) # crashed: used too much ram again 6.5 million reviews was far past my computers
    score = torch.tensor(score).unsqueeze(dim=-1)
    print(padded.size())
    print(score.size())
    return vocab, padded, hashing, score # solution found: use secound smaller dataset after discussing with proffessor - 1/7
#return hashing bc I want to test new data

def lex_order_new(hashing, dataset, padd_width):
    lex = []
    for review in dataset:
        lister = np.zeros((padd_width), dtype=np.int64)
        index = 0
        for word in review.split():
            if word in hashing:
                lister[index] = hashing[word]
                index += 1
                if index == padd_width:
                    break
        lister = torch.tensor(lister, dtype=torch.long)
        lex.append(lister)

    lex = torch.nn.utils.rnn.pad_sequence(lex, batch_first=True).long()
    return lex
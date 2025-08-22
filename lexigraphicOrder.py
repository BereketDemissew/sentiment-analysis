import torch # pip install torch torchvision torchaudio
import numpy as np

# Purpose: processes a list of text reviews, creating a numerical representation of the words in the reviews, needed for ML
# Parameters: review (list of strings)
# Return values: vocab, padded, hashing
def lex_order(reviews, score):
    reviews = [review.lower() for review in reviews]
    words = set()
    for i in reviews:
        for n in i.split():
            words.add(n)
    words = sorted(list(words))
    vocab = len(words) + 1
    hashing = {}
    order = 1
    for i in words:
        hashing[i] = order
        order += 1
    lexOrder = []

   # make padding for training
    for review in reviews:
        hashed_tokens = []
        for words in review.split():

            hashed_tokens.append(hashing[words])
        lexOrder.append(torch.tensor(hashed_tokens, dtype=torch.long))
    padded = torch.nn.utils.rnn.pad_sequence(lexOrder, batch_first=True) # crashed: used too much ram again 6.5 million reviews was far past my computers
    score = torch.tensor(score).unsqueeze(dim=-1)
    print(padded.size())
    print(score.size())
    return vocab, padded, hashing, score # solution found: use secound smaller dataset after discussing with proffessor - 1/7
#return hashing bc I want to test new data

def lex_order_new(hashing, dataset, padd_width):
    lexOrder = []
    for review in dataset:
        hashedTokens = np.zeros((padd_width), dtype=np.int64)
        index = 0
        for word in review.split():
            if word in hashing:
                hashedTokens[index] = hashing[word]
                index += 1
                if index == padd_width:
                    break
        hashedTokens = torch.tensor(hashedTokens, dtype=torch.long)
        lexOrder.append(hashedTokens)

    lexOrder = torch.nn.utils.rnn.pad_sequence(lexOrder, batch_first=True).long()
    return lexOrder

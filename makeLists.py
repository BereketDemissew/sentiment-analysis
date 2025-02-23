import csv
import torch

# File imports
import cleanDatasets

def make_lists():
    csv.field_size_limit(1048576)
    review = []
    score = []
    #stop = 1000
    check = 2
    with open('IMDB_MovieDataset.csv', encoding='utf-8') as csvfile:

        reader = csv.DictReader(csvfile)
        for row in reader:
                review.append(row['review'])  # Append review text
                if row['sentiment'] == "positive":
                    check = "positive"
                    score.append(1.0)
                else:
                    score.append(0.0)
    if check == "positive":
        review = cleanDatasets.clean_data_movies(review)

    stop = int(len(review) * 0.1)

    test_review, test_score, review, score = make_test_val(review, score, stop)
    val_review, val_score, review, score = make_test_val(review, score, stop)
    test_score = torch.tensor(test_score).unsqueeze(dim=-1)
    val_score = torch.tensor(val_score).unsqueeze(dim=-1)

    return review, score, test_review, test_score, val_review, val_score


def make_test_val(review, score, stop):
    temp_review = []
    temp_score = []
    amount = 0
    #for negative reviews
    for i in reversed(range(len(review))):
        if amount == stop//2:
            break
        if score[i] == 0.0:
            temp_review.append(review[i])
            temp_score.append(score[i])
            del score[i]
            del review[i]
            amount += 1
    #for positive reviews
    amount = 0
    for i in reversed(range(len(review))):
        if amount == stop//2:
            break
        if score[i] == 1.0:
            temp_review.append(review[i])
            temp_score.append(score[i])
            del score[i]
            del review[i]
            amount += 1

    return temp_review, temp_score, review, score
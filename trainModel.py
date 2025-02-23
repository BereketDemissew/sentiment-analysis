import torch
import torch.nn as nn

# File imports
import calculateAccuracy
import makeLinearGraphs
import checkpoints

def training_loop(model, optimizer, vocab, padding, updated_score, val_review, val_score, test_review, test_score, trainingLosses, validationLosses, previousTrainingLoops, device):
    #training loop
    batch_size = 512
    loss_function = nn.BCELoss()
    loops = 10
    # trainingLosses = []
    # validationLosses = []
    val_batch = int(batch_size * 0.8)
    print(f'beginning training {loops} loops')
    for epoch in range(loops):

        randperm = torch.randperm(len(padding))
        padding, updated_score = padding[randperm].to(device), updated_score[randperm].to(device)
        model.train()
        total_loss = 0

        for i in range(0, len(padding), batch_size):

            mini_batch = padding[i:i + batch_size]
            mini_batch_labels = updated_score[i:i + batch_size]

            pred = model(mini_batch)
            loss = loss_function(pred, mini_batch_labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        trainingLosses.append(total_loss / (len(padding) // batch_size))

        with torch.no_grad():
            val_randomperm = torch.randperm(len(val_review))
            val_review = val_review[val_randomperm]
            val_score = val_score[val_randomperm]
            model.eval()
            val_loss = 0

            for i in range(0, len(val_review), val_batch):
                mini_batch_val = val_review[i:i +  val_batch].to(device)
                mini_batch_labels_val = val_score[i:i + val_batch].to(device)

                pred = model(mini_batch_val)
                loss_val = loss_function(pred, mini_batch_labels_val)
                val_loss += loss_val.item()
            validationLosses.append(val_loss / (len(val_review) // val_batch))

        if (epoch % 10 == 9):
          print(f"{epoch + 1 + previousTrainingLoops} epochs completed -- training loss: {round(trainingLosses[-1], 2)} -- validation loss: {round(validationLosses[-1], 2)}")
    checkpoint = {
    # 'model': model.state_dict(),
    # 'optimizer': optimizer.state_dict(),
    'loops': loops + previousTrainingLoops,
    'trainingLosses': trainingLosses,
    'validationLosses': validationLosses
    }
    
    # if lowest is below 0.4 change how it works for later
    # return checkpoint
    return checkpoint

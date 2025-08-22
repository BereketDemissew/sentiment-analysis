import torch
import torch.nn as nn
import time
import datetime

# File imports
import sendEmail

def training_loop(model, optimizer, padding, updated_score, val_review, val_score, trainingLosses, validationLosses, previousTrainingLoops, device):
    #training loop
    batch_size = 512
    loss_function = nn.BCELoss()
    # epochs = 1
    val_batch = int(batch_size * 0.8)
    print('beginning training now')
    startTime = time.time()
    for epochs in range(1, 301):
        randperm = torch.randperm(len(padding)) # so model recognizes patterns from batch across the board and not each specific batch
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

        # back propogation for validation
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
        
        # checking for overfitting to stop model (loops is a counter now)
        if epochs > 100:
            currAvg = sum(validationLosses[-20:]) / 20
            prevAvg = sum(validationLosses[-50:-20]) / 30
            if currAvg > prevAvg + 0.04: # float is the buffer between the averages
                print(f'training completed {epochs} loops trained in {str(datetime.timedelta(seconds=round(int(time.time()-startTime), 2)))}')
                break

        emailRegularity = 20
        if epochs != 0: # avoid ZeroDivisionError when running 0 extra loops
            if (epochs % emailRegularity == 0):
                print(f"{epochs + previousTrainingLoops} epochs completed -- training loss: {round(trainingLosses[-1], 2)} -- validation loss: {round(validationLosses[-1], 2)}")
                # not using line anymore because training only takes 15 min
                # sendEmail.sendEmail(epochs, round(trainingLosses[-1], 2), round(validationLosses[-1], 2), False, "", accuracy = 0.0)
    checkpoint = {
    'loops': epochs + previousTrainingLoops,
    'trainingLosses': trainingLosses,
    'validationLosses': validationLosses,
    'batchSize': batch_size
    }
    return checkpoint

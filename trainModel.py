import torch
import torch.nn as nn

# File imports
import sendEmail

def training_loop(model, optimizer, padding, updated_score, val_review, val_score, trainingLosses, validationLosses, previousTrainingLoops, device):
    #training loop
    batch_size = 512
    loss_function = nn.BCELoss()
    loops = 1
    val_batch = int(batch_size * 0.8)
    print(f'beginning training {loops} loops')
    for epoch in range(loops):

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
        print('loops val:', loops)
        if loops != 0 and loops // 20 != 0: # avoid ZeroDivisionError when running 0 extra loops
            print('enters as loops != 0 and loops // 20 != 0')
            if ((epoch + 1) % (loops // 20) == 9):
                print(f"{epoch + 1 + previousTrainingLoops} epochs completed -- training loss: {round(trainingLosses[-1], 2)} -- validation loss: {round(validationLosses[-1], 2)}")
                sendEmail.sendEmail(epoch + 1, round(trainingLosses[-1], 2), round(validationLosses[-1], 2), False, "", accuracy = 0.0)
    checkpoint = {
    'loops': loops + previousTrainingLoops,
    'trainingLosses': trainingLosses,
    'validationLosses': validationLosses
    }
    return checkpoint

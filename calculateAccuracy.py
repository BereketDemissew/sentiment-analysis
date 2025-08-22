import torch
import numpy as np

def compute_accuracy(model, test_data, test_labels, device, batchSize):
    model.eval()
    accuracyList = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batchSize):
            # randperm = torch.randperm(len(test_data)) # since testing accuracy on whole set, don't need to do randperm
            # test_data, test_labels = test_data[randperm], test_labels[randperm]
            mini_batch = test_data[i : i+batchSize].to(device)
            mini_batch_labels = test_labels[i : i+batchSize].to(device)
            
            
            logits = model(mini_batch)                   # raw outputs
            probabilities = torch.sigmoid(logits)        # convert to [0,1]
            predictions = (probabilities > 0.5).long()   # threshold at 0.5
            
            correct = torch.sum(predictions.view(-1) == mini_batch_labels.view(-1))
            accuracy = correct.item()/predictions.size(0) # *100, want accuracy to be decimal to fit on loss graph
            accuracyList.append(accuracy)
    accuracyList = np.mean(accuracyList)
    accuracy = accuracyList.mean()

    print(f"accuracy = {accuracy*100:.2f}%")
    return accuracy

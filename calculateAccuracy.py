import torch
import numpy as np

def compute_accuracy(model, test_data, test_labels, epochs, device):
    model.eval()
    batch_size = int(512 * .8)
    accuracyList = []
    correct = []
    with torch.no_grad():
        for i in range(epochs):
            randperm = torch.randperm(len(test_data))
            test_data, test_labels = test_data[randperm], test_labels[randperm]
            model.eval()
            for i in range(0, len(test_data), batch_size):
                mini_batch = test_data[i:i + batch_size].to(device)
                mini_batch_labels = test_labels[i:i + batch_size].to(device)
                pred = model(mini_batch)
                predictions = torch.round(pred)
                correct = torch.sum(predictions == mini_batch_labels)

                accuracy = correct.item()/predictions.size(0) *100
                accuracyList.append(accuracy)
    accuracyList = np.array(accuracyList)
    accuracy = accuracyList.mean()

    print(f"accuracy = {accuracy:.2f}%")
    return accuracy

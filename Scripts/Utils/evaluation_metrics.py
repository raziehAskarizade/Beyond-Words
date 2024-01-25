import torch


def calculate_accuracy(model, dataloader):

    model.eval()
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(inputs)
        predicted = torch.argmax(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


def baseline_accuracy(labels):
    bincounts = torch.bincount(labels)
    accuracy = bincounts[torch.argmax(bincounts)] / len(labels)
    return accuracy

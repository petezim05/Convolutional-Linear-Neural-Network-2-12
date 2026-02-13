import ConLinNet as net
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

#set cuda as accelerator
if torch.cuda.is_available():
    loc = torch.device("cuda")
else:
    loc = torch.device("cpu")

#load and prep data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, pin_memory=True)

#criteria to run
model = net.LeCopy()
model = model.to(loc)

lossFunc =nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)

#so it begins
def trainEpoch(model, dataset, lossFunc, loc, optimizer):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in dataset:
        x = x.to(loc, non_blocking = True)
        y = y.to(loc, non_blocking = True)

        optimizer.zero_grad()
        guess = model(x)
        loss = lossFunc(guess, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        loss_sum += loss.item() * batch_size
        preds = guess.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += batch_size

    avgLoss = loss_sum / total
    accuracy = correct / total
    return avgLoss, accuracy


@torch.no_grad()
def test(model, dataset, lossFunc, loc):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in dataset:
        x = x.to(loc)
        y = y.to(loc)

        guess = model(x)
        loss = lossFunc(guess, y)

        batch_size = x.size(0)
        loss_sum += loss.item() * batch_size
        preds = guess.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += batch_size

    avgLoss = loss_sum / total
    accuracy = correct / total
    return avgLoss, accuracy


#try it
steps = 5
for epoch in range(0 , steps):
    trainingLoss, trainingAcuracy = trainEpoch(model , train_loader , lossFunc, 
                                               loc, optimizer)
    testLoss, testAccuracy = test(model , test_loader , lossFunc, loc)
    
    print(f"Epoch: {epoch}\nTraining loss: {trainingLoss}\nTraining Accuracy: {trainingAcuracy}")
    print(f"Epoch: {epoch}\nTest loss: {testLoss}\nTest Accuracy: {testAccuracy}")


#ought to be 100 lines
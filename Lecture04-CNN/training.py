import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.modules.batchnorm import BatchNorm1d
from tqdm.notebook import tqdm

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from IPython.display import clear_output

from convnet import ConvNet


train_data = datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transforms.ToTensor())

train_size = int(len(train_data) * 0.8)
val_size =  len(train_data) - train_size

train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


def show_images(images, labels):
    f, axes = plt.subplots(1, 10, figsize=(30,5))

    for i, axis in enumerate(axes):
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img)
        axes[i].set_title(labels[i].numpy())

    plt.show()


for batch in train_loader:
    images, labels = batch
    break


from sklearn.metrics import accuracy_score


def evaluate(model, dataloader, loss_fn):
    y_pred_list = []
    y_true_list = []
    losses = []

    for i, batch in enumerate(tqdm(dataloader)):
        X_batch, y_batch = batch

        with torch.no_grad():
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))
            loss = loss.item()

            losses.append(loss)

            y_pred = torch.argmax(logits, dim=1)

        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

    accuracy = accuracy_score(y_pred_list, y_true_list)

    return accuracy, np.mean(losses)


def train(model, loss_fn, optimizer, n_epoch=6):
    model.train(True)

    data = {
        'acc_train': [],
        'loss_train': [],
        'acc_val': [],
        'loss_val': []
    }

    for epoch in tqdm(range(n_epoch)):

        for i, batch in enumerate(tqdm(train_loader)):
            X_batch, y_batch = batch
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('On epoch end', epoch)

        acc_train_epoch, loss_train_epoch = evaluate(model, train_loader, loss_fn)
        print('Train acc:', acc_train_epoch, 'Train loss:', loss_train_epoch)

        acc_val_epoch, loss_val_epoch = evaluate(model, val_loader, loss_fn)
        print('Val acc:', acc_val_epoch, 'Val loss:', loss_val_epoch)

        data['acc_train'].append(acc_train_epoch)
        data['loss_train'].append(loss_train_epoch)
        data['acc_val'].append(acc_val_epoch)
        data['loss_val'].append(loss_val_epoch)

    return model, data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model, data = train(model, loss_fn, optimizer, n_epoch=6)

test_acc, test_loss = evaluate(model, test_loader, loss_fn)

print('Test_accuracy is ', test_acc)

assert test_acc >= 0.5, 'Accuracy on test < 0.5!'

model.eval()
x = torch.randn((1, 3, 32, 32))
torch.jit.save(torch.jit.trace(model.cpu(), (x)), "model.pth")

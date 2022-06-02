from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
from custom_dataset import CustomDataset

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tfms = T.Compose([
    T.Resize((168,168)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(*imagenet_stats,inplace=True),
    T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
    T.Resize((168,168)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])

#load datasets
train_dataset = CustomDataset('/data/datasets/esc50/esc50_processed/meta/esc50.csv', '/data/datasets/esc50/esc50_processed_2/images/',transform=train_tfms)
test_dataset = CustomDataset('/data/datasets/esc50/esc50_processed/meta/esc50.csv','/data/datasets/esc50/esc50_processed_2/images', transform=valid_tfms)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#define model
net = models.resnet18(pretrained=True)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

#redefine the last layer
num_ftrs = net.fc.in_features
num_classes = 50
net.fc = nn.Linear(num_ftrs, 50)
net.fc = net.fc.to(device)

#basic training script
n_epochs = 1
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        outputs = net(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(test_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    net.train()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
from util_functions import make_mask, train, test, prune_by_percentile, current_pruned_percent, copy_params, extract_non_zero_params,save_model_distribution

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=True)

model = fc1.fc1().to(device)
original_params = copy_params(model)

mask = make_mask(model)
save_model_distribution(model, mask, 'distributions/original.txt')

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
criterion_test = nn.CrossEntropyLoss(reduction='sum')

epochs = 10
prune_percent = .35

for epoch in range(1, epochs + 1):
    train(model, mask, train_loader, optimizer, criterion, epoch, DEBUG=True)
    test(model, test_loader, criterion_test, epoch, DEBUG=True)
    
    if epoch % 1 == 0:
        prune_by_percentile(model, mask, prune_percent*100)
        print("Epoch: {}, Currently pruned to {:.2f}% capacity\n".format(epoch, current_pruned_percent(model)*100))
        save_model_distribution(model, mask, 'distributions/after_epoch_{}.txt'.format(epoch))

final_params = copy_params(model)


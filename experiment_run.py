import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dprint(s, DEBUG=True):
    if DEBUG:
        print(s)

def create_output_folder(dataset, arch_type, trial_num):
    out_dir = 'outputs'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = out_dir + '/{}'.format(dataset)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = out_dir + '/{}'.format(arch_type)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = out_dir + '/trial_{}'.format(trial_num)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    return out_dir

def dataset_and_model(dataset, batch_size, arch_type):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet

    elif dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet
    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=True)

    model = None
    if arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif arch_type == "LeNet5":
        model = LeNet5.LeNet5().to(device)
    elif arch_type == "AlexNet":
        model = AlexNet.AlexNet().to(device)
    elif arch_type == "vgg":
        model = vgg.vgg16().to(device)
    elif arch_type == "resnet":
        model = resnet.resnet18().to(device)
    elif arch_type == "densenet":
        model = densenet.densenet121().to(device)
    else:
        print("\nWrong Model choice\n")
        exit()

    return train_loader, test_loader, model

def save_distribution(model, out_file):
    dprint("\tSaved distribution")
    with open(out_file, 'wb') as f:
        np.savetxt(f, extract_non_zero_params(copy_params(model)))

def train_fully(model, mask, num_epochs, train_loader, test_loader, optimizer, criterion, criterion_sum, save_distribution_frequency, out_dir, header):
    dprint("\nTraining model")

    train_loss = np.zeros(num_epochs+1)
    test_loss = np.zeros(num_epochs+1)
    test_acc = np.zeros(num_epochs+1)

    train_loss[0], _ = get_loss_and_acc(model, train_loader, criterion_sum)
    test_loss[0], test_acc[0] = get_loss_and_acc(model, test_loader, criterion_sum)

    save_distribution(model, '{}/{}_dist_0.txt'.format(out_dir,header))

    for epoch in np.arange(1,num_epochs+1):
        train(model, mask, train_loader, optimizer, criterion)
        train_loss[epoch], _ = get_loss_and_acc(model, train_loader, criterion_sum)
        test_loss[epoch], test_acc[epoch] = get_loss_and_acc(model, test_loader, criterion_sum)

        if epoch != num_epochs and epoch % save_distribution_frequency == 0:
            save_distribution(model, '{}/{}_dist_{}.txt'.format(out_dir,header,epoch))
        
        if epoch % 20 == 0:
            dprint("\tFinished epoch {} with train_loss = {}, test_loss = {}, test_acc = {}".format(train_loss[epoch], test_loss[epoch], test_acc[epoch]))

    save_distribution(model, '{}/{}_dist_{}.txt'.format(out_dir,header,num_epochs))

    with open('{}/{}_test_acc.txt'.format(out_dir,header),'wb') as f:
        np.savetxt(f, test_acc)
    with open('{}/{}_test_loss.txt'.format(out_dir,header),'wb') as f:
        np.savetxt(f, test_loss)
    with open('{}/{}_train_loss.txt'.format(out_dir,header),'wb') as f:
        np.savetxt(f, train_loss)
    dprint("\tSaved loss and accuracy curves")
    dprint("\tFinished Training Model")

def main(args, trial_num):
    train_loader, test_loader, model = dataset_and_model(args.dataset, args.batch_size, args.arch_type)
    dprint("\nLoaded data and model")
    
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion_sum = nn.CrossEntropyLoss(reduction='sum')
    
    out_dir = create_output_folder(args.dataset, args.arch_type, trial_num)
    dprint("\nCreated outdir: {}".format(out_dir))

    dprint("\nCreating pruned models")
    model_pruning_trainer = lambda model, mask: train(model, mask, train_loader, optimizer, criterion)
    pruned_models = create_pruned_models(model, model_pruning_trainer, 100*args.pruning_s, args.pruning_j, args.pruning_n)
    dprint("Created pruned models")

    experiment_runner = lambda model, mask, out_dir, header: train_fully(model, mask, args.num_epochs, train_loader, test_loader, optimizer, criterion, criterion_sum, args.save_dist_freq, out_dir, header)

    mask_0, theta_0 = pruned_models[0]
    D_0 = extract_non_zero_params(theta_0)

    reinitialize_model(model, theta_0, mask_0)
    dprint("\nReinitialized with theta_0, full. Starting run")
    experiment_runner(model, mask_0, out_dir, '0_full_Dinit')

    for n in args.pruning_n:
        dprint("\nStarting new n: {}".format(n))
        mask_n, theta_n = pruned_models[n]
        D_n = extract_non_zero_params(theta_n)

        #pruned, D_init
        reinitialize_model(model, theta_0, mask_n)
        dprint("\nReinitialized with theta_0, pruned. Starting run")
        experiment_runner(model, mask_n, out_dir, '{}_pruned_Dinit'.format(n))

        #pruned, D_n
        reinitialize_model_sample(model, D_n, mask_n)
        dprint("\nReinitialized with D_n, pruned. Starting run")
        experiment_runner(model, mask_n, out_dir, '{}_pruned_Dn'.format(n))

        #pruned, D_0
        reinitialize_model_sample(model, D_0, mask_n)
        dprint("\nReinitialized with D_0, pruned. starting run")
        experiment_runner(model, mask_n, out_dir, '{}_pruned_D0'.format(n))

        #full, D_n
        reinitialize_model_sample(model, D_n, mask_0)
        dprint("\nReinitialized with D_n, full. starting run")
        experiment_runner(model, mask_0, out_dir, '{}_full_Dn'.format(n))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--arch_type', type=str, default='fc1')
    parser.add_argument('--pruning_s', type=float, default=.20)
    parser.add_argument('--pruning_j', type=int, default=5)
    parser.add_argument('--pruning_n', type=int, nargs='+', default=[2,5,8,10])
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--save_dist_freq', type=int, default=75)

    args = parser.parse_args()

    for trial_num in range(1, args.num_trials+1):
        main(args, trial_num)




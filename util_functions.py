import numpy as np
import torch

def dprint(s, DEBUG=True):
    if DEBUG:
        print(s)

def get_loss_and_acc(model, data_loader, criterion_sum):
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion_sum(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = 100. * correct / len(data_loader.dataset)
    return loss, acc

def train(model, mask, train_loader, optimizer, criterion, DEBUG=False):
    ZERO_VAL = 1e-5
    
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    loss = None

    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()

        for name, p in model.named_parameters():
            if 'weight' in name:
                grad = p.grad.data.cpu().numpy()
                p.grad.data = torch.from_numpy(grad * mask[name]).to(p.device)
        optimizer.step()
        
        if DEBUG and batch_idx >= 100:
            break

def make_mask(model):
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[name] = np.ones_like(tensor)

    return mask

def prune_by_percentile(model, mask, percent):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy()
            percentile_value = np.percentile(abs(weights[mask[name]!=0]), percent)

            new_mask = np.where(abs(weights) < percentile_value, 0, mask[name])
            param.data = torch.from_numpy(weights * new_mask).to(param.device)
            mask[name] = new_mask

def current_pruned_percent(model):
    total_params = 0
    total_active = 0

    for name, p in model.named_parameters():
        if 'weight' in name:
            weights = p.data.cpu().numpy()
            total_active += np.count_nonzero(weights)
            total_params += weights.size
    return total_active/total_params

def copy_params(model):
    params = {}
    for name, p in model.named_parameters():
        if 'weight' in name:
            params[name] = np.copy(p.data.cpu().numpy())
    return params

def extract_non_zero_params(weights):
    all_params = np.array([])
    for key in weights:
        layer = weights[key]
        all_params = np.concatenate((all_params, layer[np.nonzero(layer)]))
    return all_params

def reinitialize_model(model, weights, mask):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = torch.from_numpy(weights[name] * mask[name]).to(param.device)

def reinitialize_model_sampling(model, weights, mask):
    for name, param in model.named_parameters():
        if 'weight' in name:
            new_weights = np.random.choice(weights, size=param.data.cpu().numpy().shape)
            param.data = torch.from_numpy(new_weights * mask[name]).to(param.device)

def create_pruned_models(model, model_trainer, pruning_s, pruning_j, pruning_n):
    mask = make_mask(model)

    initial_params = copy_params(model)
    dprint("\tCreated initial model and mask")
    ret = {}

    ret[0] = (mask, initial_params)
    
    max_pruning = max(pruning_n)
    for n in range(1, max_pruning+1):
        dprint("\tStarting pruning n={}".format(n))
        for j in range(1,pruning_j+1):
            dprint("\t\tTraining epoch: {}".format(j))
            model_trainer(model, mask)
        dprint("\tPruning the model")
        prune_by_percentile(model, mask, pruning_s)
        
        if n in pruning_n:
            print("\tSaving pruned model")
            ret[n] = (mask, copy_params(model))

        dprint("\tReinitializing masked model")
        reinitialize_model(model, initial_params, mask)

    return ret



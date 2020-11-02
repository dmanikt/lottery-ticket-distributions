import numpy as np
import torch

def test(model, test_loader, criterion, epoch, DEBUG=False):
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if DEBUG:
        print('\nTest Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

def train(model, mask, train_loader, optimizer, criterion, epoch, DEBUG=False):
    ZERO_VAL = 1e-5
    
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    loss = None

    if DEBUG:
        print("Starting Train Epoch: {}".format(epoch))

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
        
        if DEBUG and batch_idx >= 200:
            continue

        if DEBUG and batch_idx % 20 == 0:
            print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()

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
            tensor = param.data.cpu().numpy()
            percentile_value = np.percentile(abs(tensor[mask[name]!=0]), percent)

            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[name])

            param.data = torch.from_numpy(tensor * new_mask).to(param.device)
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

def extract_non_zero_params(model,mask):
    named_params = copy_params(model)
    all_params = np.array([])
    for key in named_params:
        masked_layer = named_params[key] * mask[key]
        all_params = np.concatenate((all_params, masked_layer[np.nonzero(masked_layer)]))
    return all_params

def save_model_distribution(model, mask, filename):
    all_params = extract_non_zero_params(model, mask)
    with open(filename,'wb') as f:
        np.save(f,all_params)

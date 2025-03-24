import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear


class GCN_model(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(GCN_model, self).__init__()

        self.conv1 = GCNConv(2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # set number of class here
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin(x)

        return x

def GCN(hidden_channels, num_classes):
    model = GCN_model(hidden_channels, num_classes)

    return model

def train(model, train_loader, optimizer, criterion):
    model.train()

    for data in train_loader:
        x = data.x.cuda()
        y = data.edge_index.cuda()
        z = data.batch.cuda()
        out = model(x, y, z)
        loss = criterion(out, torch.LongTensor(data.y).cuda()) # Compute loss
        loss.backward() # Derive gradients
        optimizer.step() # Update parameters based on gradients
        optimizer.zero_grad() # Clear gradients

    return model

def train_accuracy(model, train_loader):
    model.eval()
    correct = 0
    for data in train_loader:
        x = data.x.cuda()
        y = data.edge_index.cuda()
        z = data.batch.cuda()
        out = model(x, y, z)
        pred = out.argmax(dim = 1).cpu()
        correct += int((pred == torch.LongTensor(data.y)).sum())
    epoch_acc = correct / len(train_loader.dataset)

    return epoch_acc

def val(model, val_loader, criterion, best_loss, model_save_loc):
    model.eval()
    correct = 0
    loss = 0
    for data in val_loader:
        x = data.x.cuda()
        y = data.edge_index.cuda()
        z = data.batch.cuda()
        out = model(x, y, z)
        pred = out.argmax(dim = 1).cpu()
        correct += int((pred == torch.LongTensor(data.y)).sum())
        loss += criterion(out.cpu(), torch.LongTensor(data.y)).item() * y.size(0)
    epoch_acc = correct / len(val_loader.dataset)
    epoch_loss = loss / len(val_loader.dataset)
    
    # if validation loss is less than best loss,
    # save the model.
    # meaning that the model with the least loss will be saved to model_save_loc
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), model_save_loc)

    return epoch_acc, epoch_loss, best_loss

def test(hidden_channels, num_classes, test_loader, model_save_loc):
    model = GCN_model(hidden_channels, num_classes)
    model.load_state_dict(torch.load(model_save_loc))
    model.cuda()
    model.eval()
    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        x = data.x.cuda()
        y = data.edge_index.cuda()
        z = data.batch.cuda()
        out = model(x, y, z)
        pred = out.argmax(dim=1).cpu()  # Use the class with highest probability.
        correct += int((pred == torch.LongTensor(data.y)).sum())  # Check against ground-truth labels.
        epoch_acc = correct / len(test_loader.dataset)

    return epoch_acc  # Derive ratio of correct predictions.

def test_pred(hidden_channels, num_classes, test_loader, model_location):
    model = GCN_model(hidden_channels, num_classes)
    model.load_state_dict(torch.load(model_location))
    model.cuda()
    model.eval()
    for data in test_loader:
        x = data.x.cuda()
        y = data.edge_index.cuda()
        z = data.batch.cuda()
        out = model(x, y, z)
        prob = (F.softmax(out, dim=1).cpu().data.numpy()).tolist()[0]
        pred = out.argmax(dim=1).cpu()

    return pred.item(), prob[pred.item()], prob
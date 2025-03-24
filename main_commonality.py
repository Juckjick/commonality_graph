import os
import random
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torchmetrics import AUROC, Accuracy, ConfusionMatrix
from utils import DMN_top_label_conf_extraction

import Model.GCN as GCN

def main(seed_num):
    # setting seed number
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)

    organs = ['stomach', 'colon']
    for organ in organs:
        graph_path = 'files/graphs/' # path to saved graphs

        model_path = 'models/trained_test/' # path to save a trained model
        if not os.path.exists(os.path.join(model_path, organ)):
            os.makedirs(os.path.join(model_path, organ))
        model_save_path = os.path.join(model_path, organ, f'{seed_num}.pkl')

        # training models
        print('Training model ...')
        train_data, val_data, test_data = get_data(graph_path, organ)
        training_model(train_data, val_data, test_data, model_save_path)

        # testing a trained model
        print('Test model ...')
        f = open(graph_path + f'{organ}_test_graphs.b', 'rb')
        slide_names, slide_gts, all_graphs = pickle.load(f)
        test_model(organ, seed_num, slide_names, slide_gts, all_graphs, model_save_path)

def get_data(file_path, organ):
    modes = ['train', 'val', 'test']
    train_data = []
    val_data = []
    test_data = []
    for mode in modes:
        path = os.path.join(file_path, f'{organ}_{mode}_graphs.b')
        f = open(path, 'rb')
        _, _, all_graphs = pickle.load(f)
        for i in range(len(all_graphs)):
            features = DMN_top_label_conf_extraction(all_graphs[i].x.tolist())
            features = torch.tensor(features, dtype = torch.float)
            data = Data(x = features, edge_index = all_graphs[i].edge_index, y = all_graphs[i].y)

            if mode == 'train':
                train_data.append(data)
            elif mode == 'val':
                val_data.append(data)
            else:
                test_data.append(data)
    
    return train_data, val_data, test_data

def training_model(train, val, test, model_save_path):
    # displaying the number of graphs
    print(f"Data info: #training:{len(train)}\t#val: {len(val)}\t#test: {len(test)}")

    # loading data to DataLoader
    train_loader = DataLoader(train, batch_size = 64, shuffle = True)
    val_loader = DataLoader(val, batch_size = 64, shuffle = False)
    test_loader = DataLoader(test, batch_size = 64, shuffle = False)

    # setting model's hyperparameter
    hidden_channels = 64
    num_classes = 3
    model = GCN.GCN(hidden_channels = hidden_channels, num_classes = num_classes)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # training a model
    best_loss = 1000.0
    for epoch in range(1, 200):
        model = GCN.train(model, train_loader, optimizer, criterion)
        train_acc = GCN.train_accuracy(model, train_loader)
        val_acc, val_loss, best_loss = GCN.val(model, val_loader, criterion, best_loss, model_save_path) # a model is saved in <GCN.val> function
        if (epoch+1)%10 == 0:
            print(f"epoch: {epoch:03d}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}, val loss: {val_loss:.4f}")

    # # (optional) passing the saved model and test data to measure the accuracy of the model.
    # test_acc = GCN.test(hidden_channels, num_classes, test_loader, model_save_path)
    # print(f'Test Acc: {test_acc:.4f}')

def test_model(organ, seed_num, slide_names, slide_gts, all_graphs, model_path):
    preds = []
    probs = []
    for idx in range(len(slide_names)):
        graph = all_graphs[idx]

        # preparing test data and loading them to DataLoader
        hidden_channels = 64
        num_classes = 3
        features = DMN_top_label_conf_extraction(graph.x.tolist())
        features = torch.tensor(features, dtype = torch.float)
        data = Data(x = features, edge_index = graph.edge_index)
        test_loader = DataLoader([data], batch_size = 1, shuffle = False)

        # testing the saved model 
        pred, _, prob = GCN.test_pred(hidden_channels, num_classes, test_loader, model_path)
        preds.append(pred)
        probs.append(prob)

    # getting models' performances: accuracy, confusion matrix, and AUROC
    accuracy = Accuracy(task = 'multiclass', num_classes = 3)
    acc_torch = accuracy(torch.tensor(preds), torch.tensor(slide_gts))
    confusionmatrix = ConfusionMatrix(task = 'multiclass', num_classes = 3)
    conf_torch = confusionmatrix(torch.tensor(preds), torch.tensor(slide_gts))
    auroc = AUROC(task = 'multiclass', num_classes = 3)
    auroc_torch = auroc(torch.tensor(probs), torch.tensor(slide_gts))

    print(f'organ: {organ}, seed: {seed_num}, accuracy_t: {acc_torch}, auroc_t: {auroc_torch}, conf_t: {conf_torch}')

if __name__ == '__main__':
    seeds = [32, 60, 42, 2023, 224]
    for seed_num in seeds:
        main(seed_num)
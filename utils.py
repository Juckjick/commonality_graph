import math
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment

# main function of WSI-level graph generation 
def wsi_generator(conf_list, label_list, loc_list, slide_label):
    X = 256 
    Y = 256
    neighbors_excluding_diagonal = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                                for y2 in range(y-1, y+2)
                                if (-1 < x <= X and -1 < y <= Y and
                                    (x != x2 or y != y2) and (x != x2-1 or y != y2-1) and
                                    (x-1 != x2 or y-1 != y2) and (x == x2 or y == y2) and
                                    (0 <= x2 <= X) and (0 <= y2 <= Y))]

    # getting cardinal-connected nodes information
    source = []
    dest = []
    for node in loc_list:
        neighbors = neighbors_excluding_diagonal(node[0], node[1])
        for n in neighbors:
            if list(n) in loc_list:
                source.append(loc_list.index(node))
                dest.append(loc_list.index(list(n)))
    edge_index = torch.tensor([source, dest], dtype = torch.long)

    x = [] 
    for i in range(len(conf_list)):
        t1 = label_list[i]
        t2 = conf_list[i]
        arr = [t1, t2]
        x.append(arr)
    x = torch.Tensor(x) # nodes' features
    y = slide_label # slide's groundtruth
    location = loc_list
    data = Data(x = x, edge_index = edge_index, y = y, location = location)

    return data

# functions for slice-level graph generation
def find_centroids(components, location_list):  
    centroids = []
    for _, g in enumerate(components):
        location = [location_list[n] for n in list(g.nodes())]
        centroid = [sum(x)/len(x) for x in zip(*location)]
        centroids.append(centroid)
    return centroids

def find_min_dist(centroids):
    d_min = 1000
    com1 = 0
    com2 = 0
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if j > i:
                x1 = centroids[i][0]
                y1 = centroids[i][1]
                x2 = centroids[j][0]
                y2 = centroids[j][1]
                distance = math.hypot(x2 - x1, y2 - y1)
                if distance < d_min:
                    d_min = distance
                    com1 = i
                    com2 = j
    return [com1, com2], d_min

def get_sl_graphs(components, location_list):
    distance = 0
    result = components
    temp = components
    
    # applying the center of mass theory to compose slice-level graphs
    # if the distance between components is still less than 45,
    # the components is connected as one bigger graph 
    # and calculate new centroid of the new one.
    while(distance < 45):
        centroids = find_centroids(temp, location_list)
        com_list, distance = find_min_dist(centroids)

        if distance > 45:
            break
    
        arr = []
        for i in range(len(temp)):
            if i not in com_list:
                arr.append(temp[i])
        arr.append(nx.compose(temp[com_list[0]], temp[com_list[1]]))
        
        result = arr
        temp = arr

    return result

# main function of slice-level graph generation
def slg_generator(ws_graph):
    sd = pd.DataFrame(np.transpose(ws_graph.edge_index.numpy()), columns = ["source", 'dest'])
    g = nx.from_pandas_edgelist(sd, source = 'source', target = 'dest')
    components = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    location_list = ws_graph.location

    slice_lv_graphs = get_sl_graphs(components, location_list)

    return slice_lv_graphs

# functions for commonality graph construction
def DMN_top_label_conf_extraction(features):
    results = []
    for i in range(len(features)):
        label = features[i][0]
        conf = features[i][1]
        if int(label[0]) == 0:
            results.append(['D'] + [max(conf)])
        elif int(label[0]) == 1:
            results.append(['M'] + [max(conf)])
        elif int(label[0]) == 2:
            results.append(['N'] + [max(conf)])
    return results

def HEOM(key, value1, value2):
    if key == 'label':
        return 0 if value1.upper() == value2.upper() else 1   
    elif key == 'degree':
        degree_range = 7 # max_deg = 8, min_deg = 1
        cost = abs(value1 - value2) / degree_range
        return cost
    elif key.startswith('conf'):
        degree_range = 1 # max_deg = 1, min_deg = 0
        cost = abs(value1 - value2) / degree_range
        return cost
    
def signature_setter(graph, features):
    node_signatures = {}
    for node, data in sorted(graph.nodes(data = True)):
        node_signatures[node] = {'label': features[node][0], 
                                 'conf': features[node][1],
                                 'degree': graph.degree(node)
        }
    
    return node_signatures

# main function of commonality graph construction
def commonality_g_constructor(components, all_features):
    graph_arr = components # [G, H, I] # components
    features = DMN_top_label_conf_extraction(all_features)

    # step1: setting node signatures
    graph_ns_arr = []
    for graph in graph_arr:
        ns_arr = signature_setter(graph, features)
        graph_ns_arr.append(ns_arr)

    # step2: calculating cost matrix
    cost_matrix_arr = []
    for idx in range(len(graph_ns_arr)-1):
        first_ns = graph_ns_arr[idx]
        second_ns = graph_ns_arr[idx+1]
        cost_matrix = []
        for i in first_ns:
            cost_per_node = []
            for j in second_ns:
                cost = 0
                for v in second_ns[j]:
                    cost += HEOM(v, first_ns[i][v], second_ns[j][v])         
                cost_per_node.append(cost)
            cost_matrix.append(cost_per_node)
        cost_matrix_arr.append(cost_matrix)

    # step3: permutation matrix
    cost_idx_arr = []
    row_idx_arr = []
    for cost_matrix in cost_matrix_arr:
        np_cost_matrix = np.array(cost_matrix)
        row_idx, col_idx = linear_sum_assignment(np_cost_matrix)
        cost_idx_arr.append(col_idx)
        row_idx_arr.append(row_idx)
        
    # step 4: compose componants to a bigger graph
    X = nx.union_all(graph_arr)
    size_ = 0
    size_fg = len(row_idx_arr[0]) # length of the first graph
    for i, (c, r) in enumerate(zip(cost_idx_arr, row_idx_arr)):
        for index, data in enumerate(c):
            edge1 = size_ + r[index]
            edge2 = size_ + size_fg + data
            X.add_edge(edge1, edge2)
        size_ += len(c)

    return X

# function for data transformation
def data_transform(transform_g, whole_g):
    # getting edge_index
    adj = nx.to_scipy_sparse_matrix(transform_g).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim = 0)

    # getting features
    features = [whole_g.x[f].tolist() for f in list(transform_g.nodes())]
    features = torch.tensor(features, dtype = torch.float)

    # loading to data form
    data = Data(x = features, edge_index = edge_index, y = whole_g.y)

    return data
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import networkx as nx

parent=[]
rank=[]

def knnGraph(data,k):
    kdtree=KDTree(data)
    distance,indices=kdtree.query(data,k=k+1)
    nearest_distance = distance[:, 1:]
    nearest_indices=indices[:,1:]
    #np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    #print(nearest_indices)
    G=nx.Graph()
    G_nomutual=nx.Graph()
    DG=nx.DiGraph()
    adj_matrix=np.zeros((len(data),len(data)))
    adj_matrix_nomutual=np.zeros((len(data),len(data)))
    for i in range(len(data)):
        DG.add_node(i,pos=data[i])
    for i in range(len(data)):
        for j in range(k):
            weight=np.exp(-nearest_distance[i][j]**2/2)
            DG.add_edge(i,nearest_indices[i][j],weight=weight)
            if  G_nomutual.has_edge(i,nearest_indices[i][j]):
                continue
            G_nomutual.add_edge(i,nearest_indices[i][j],weight=weight)
            adj_matrix_nomutual[i][nearest_indices[i][j]]=G_nomutual[i][nearest_indices[i][j]]['weight']
            adj_matrix_nomutual[nearest_indices[i][j]][i]=G_nomutual[i][nearest_indices[i][j]]['weight']
    #print(G)
    for i in range(len(data)):
        for j in range(i,len(data)):
            if(DG.has_edge(i,j) and DG.has_edge(j,i)):
                weight=DG.get_edge_data(i,j)
                #adj_matrix[i][j] = DG.get_edge_data(i, j).get('weight', 1.0)
                adj_matrix[i][j]=DG[i][j]['weight']
                adj_matrix[j][i]=DG[i][j]['weight']
                G.add_edge(i,j,weight=weight)
    return DG,G,G_nomutual,adj_matrix,adj_matrix_nomutual

def initialSet(data):
    global rank
    rank=[1 for i in range(len(data))]
    return list(i for i in range(len(data)))

def find(x):
    global parent
    if(x==parent[x]):
        return x
    parent[x]=find(parent[x])
    return parent[x]

def union(x, y):
    global parent
    global rank
    x = find(x)
    y = find(y)
    if x == y:
        return False

    # Union by rank (fix: previously failed to attach when rank[x] < rank[y])
    if rank[x] > rank[y]:
        parent[y] = x
    elif rank[x] < rank[y]:
        parent[x] = y
    else:
        parent[x] = y
        rank[y] += 1
    return True
    

def buildMolList(data, G):
    """Build initial molecule labels via connected components on mutual-kNN graph."""
    global parent
    parent = initialSet(data)

    # 遍历图中的每条边
    for i, j in G.edges():
        union(i, j)

    # Ensure every node points to its final representative (critical for correctness)
    parent = [find(i) for i in range(len(parent))]
    return parent
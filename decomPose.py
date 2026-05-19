import numpy as np
from matrixTool import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numpy import nonzero
from rouletteWheelSelection import *

def getPE(eignvalues):
    values=[1/element for element in eignvalues]
    return values

def decomCluster(res,index,label):
    label=np.array(label)
    point=res[index].getPoint()
    res_new=[]
    cluster=res[index]
    for i in range(len(res)):
        if i==index:
            continue
        res_new.append(res[i])
    cluster_1=Cluster()
    cluster_2=Cluster()
    for i in range(len(point)):
        if label[i]==0:
            cluster_1.addPoint(point[i])
        else:
            cluster_2.addPoint(point[i])
    res_new.append(cluster_1)
    res_new.append(cluster_2)
    return res_new

def decomPose(data, res, adjMatrix):
    lysm = []
    second_smallest_eigenvalues = []
    second_smallest_eigenvectors = []
    valid_indices = []  # 存储有效的原始索引
    
    # 第一步：收集所有有效簇的信息
    for i in range(len(res)):
        if adjMatrix[i].shape[0] == 1:
            continue  # 跳过大小为1的簇
        valid_indices.append(i)  # 记录有效索引
        lysm.append(normalizedLaplacian(adjMatrix[i]))
        second_smallest_eigenvalues.append(getSecondSmallestEigenvalue(lysm[-1]))
        second_smallest_eigenvectors.append(getSecondSmallestEigenvector(lysm[-1]))
    
    # 如果没有有效簇，直接返回原始结果
    if not valid_indices:
        return res
    
    PE = getPE(second_smallest_eigenvalues)
    pe_index = randWheelSelection(PE)  # 在有效簇中选择
    
    # 获取对应的原始索引
    original_index = valid_indices[pe_index]
    
    # 使用对应的原始索引获取邻接矩阵和特征向量
    pointVector = getPointVector(adjMatrix[original_index], second_smallest_eigenvectors[pe_index])
    
    # 进行聚类
    cluster = KMeans(n_clusters=2)
    cluster.fit(pointVector.reshape(-1, 1))
    label_pred = cluster.labels_
    
    # 使用原始索引分解簇
    res_new = decomCluster(res, original_index, label_pred)
    return res_new




# def decomPose(data,res,adjMatrix):#adj是每个簇的邻接矩阵
#     #for i in range(len(adjMatrix)):
#         #print(np.allclose(adjMatrix[i],adjMatrix[i],rtol=1e-5,atol=1e-8))
#     lysm=[]
#     second_smallest_eigenvalues=[]
#     second_smallest_eigenvectors=[]
#     skip=0
#     PE=[]
#     for i in range(len(res)):
#         if(adjMatrix[i].shape[0]==1):
#             skip+=1
#             continue
#         lysm.append(normalizedLaplacian(adjMatrix[i]))
#         second_smallest_eigenvalues.append(getSecondSmallestEigenvalue(lysm[i-skip]))
#         second_smallest_eigenvectors.append(getSecondSmallestEigenvector(lysm[i-skip]))
#         #PE.append(res[i-skip].getPE())
#     PE=getPE(second_smallest_eigenvalues)
#     index=randWheelSelection(PE)
#     pointVector=getPointVector(adjMatrix[index+skip],second_smallest_eigenvectors[index])
#     #print(adjMatrix[index].shape[0])
#     #print(pointVector)
#     #print(adjMatrix[index])
#     #print(lysm[index])
#     #print(second_smallest_eigenvectors[index])
#     cluster=KMeans(n_clusters=2)
#     cluster.fit(pointVector.reshape(-1,1))
#     #cluster.fit(second_smallest_eigenvectors[index].reshape(-1,1))
#     label_pred=cluster.labels_
#     res_new=decomCluster(res,index,label_pred)
#     #print(label_pred)
#     centers=cluster.cluster_centers_
#     return res_new
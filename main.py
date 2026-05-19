from graphTool import *
import argparse
import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import array,nonzero
from Cluster import *
from decomPose import *
from synThesis import *
from rouletteWheelSelection import *
from Container import *
from compoundDataset import *
from flameDataset import *
from jainDataset import *
from SpiralDataset import *
from random import randint
warnings.filterwarnings("ignore", category=RuntimeWarning)


#默认RB4<5与sys=0.8，k=7

#adjust_rand_score
# dataname='Aggregation.txt' #0.99
# dataname='Compound.txt' #0.989  RB4<=5,合并参数0.2
dataname='diamond9.txt'#0.989
# dataname='Flame.txt'#0.93
# dataname='Jain.txt' #1.0
# dataname='R15.txt'#0.98 RB<=5
# dataname='Spiral.txt' #1.0

sys=0.8
df=pd.read_csv('./datasets/'+dataname,sep='\t',header=None)
#print(df)
data=df.iloc[:, :-1]
labels_true=df.iloc[:,-1]
nmi_scores=[]
ari_scores=[]
fmi_scores=[]
attribute=len(list(data))
# scaler = MinMaxScaler(
# data_scaled = scaler.fit_transform(data))
data_scaled=np.array(data)

#seed=randint(0,200)
#print(seed)
# 创建Compound数据集生成器

#spiral_gen = SpiralDatasetRNG(seed, scale_factor=10.0)
    
#data_scaled, labels_true = spiral_gen.generate_spiral(n_samples=600, n_spirals=3)


def darw_cluster(dataset,labels_pred):
    if attribute>2:
        dataset=PCA(n_components=2).fit_transform(dataset)
    else:
        dataset=np.array(dataset)
    label=np.array(labels_pred)
    count=18
    colors=["#FF0000","#0000FF","#FFF000","#000000","#888A99","#A923C4",
            "#30AA4A","#301897","#8D2559","#124FAA","#D8ADAD","#D6DFCD",
            "#C3E9A0","#5ECCB1","#BC5FD8","#C031316E","#21941D76","#901ABE6C"]
    for i in range(int(count)):
        print(len(dataset[np.where(label==i)]))
        plt.scatter(dataset[nonzero(label==i),0],dataset[nonzero(label==i),1],c=colors[i],s=7,marker='o')
    plt.show()


def initSet(molList,molSet):
    res=[]
    for index in molSet:
        cluster=Cluster()
        for i in range(len(molList)):
            if molList[i]==index:
                cluster.addPoint(i)
        res.append(cluster)             
    return res

def updateCluster(res):
    graph=[]
    adjmatrix=[]
    for i in range(len(res)):
        res[i].initialClutser(data_scaled,A_nomutual)
        graph.append(res[i].G)
        adjmatrix.append(res[i].adjmatrix)
    return res,graph,adjmatrix

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ACROC reproduction runner")
    parser.add_argument("--no-plot", action="store_true", help="disable matplotlib plotting")
    args, _ = parser.parse_known_args()

    #cluster=KMeans(n_clusters=k)
    #cluster.fit(data)
    #labels_pred=cluster.labels_
    #centers=cluster.cluster_centers_
    #darw_cluster(data,labels_pred,centers)
    DG,G,G_nomutual,A,A_nomutual=knnGraph(data_scaled,k=7)#得到初始的无向图与邻接矩阵
    molList=buildMolList(data_scaled,G)#得到并查集,接下来的问题是如何将同一个分支放在一个集合中
    molSet=set(molList)
    res=initSet(molList,molSet)#获得初始的分类完成的集合
    graph=[]
    adjmatrix=[]
    sizeList=[]
    # for i in range(len(res)):
    #     print(res[i].count)
    # print("___________________")
    for i in range(len(res)):
        sizeList.append(res[i].count)
    sizeList.sort()
    sizeThreHold=max(sizeList[int(np.floor(len(sizeList) * sys))], 4)
    # sizeThreHold=4
    i = 0
    while i < len(res):
        if res[i].count <= sizeThreHold:
            res_temp=reassignCluster(res, i, DG)
            if len(res)!=len(res_temp):
                res=res_temp
                # i=0
            else:
                i+=1
        else:
            i+=1 
    # for i in range(len(res)):
        # print(res[i].count)
    res,graph,adjmatrix=updateCluster(res)
    res,graph,adjmatrix,labels_pred=container(data_scaled,res,graph,adjmatrix,A_nomutual,G_nomutual,DG,tol=5,check_interval=10,max_iter=200,labels_true=labels_true)
    ari_scores=adjusted_rand_score(labels_true,labels_pred)
    nmi_scores=normalized_mutual_info_score(labels_true,labels_pred)
    fmi_scores=fowlkes_mallows_score(labels_true,labels_pred)
    # for i in range(len(res)):
    #      print(res[i].count)
    print(f"ARI={float(ari_scores)}")
    print(f"NMI={float(nmi_scores)}")
    print(f"FMI={float(fmi_scores)}")
    if not args.no_plot:
        darw_cluster(data_scaled,labels_pred)

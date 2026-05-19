import numpy as np
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from tqdm import tqdm
from Cluster import *
from synThesis import *
from decomPose import *

def updateCluster(cluster,data_scaled,A_nomutual):
    graph=[]
    adjmatrix=[]
    for i in range(len(cluster)):
        cluster[i].initialClutser(data_scaled,A_nomutual)
        graph.append(cluster[i].G)
        adjmatrix.append(cluster[i].adjmatrix)
    return cluster,graph,adjmatrix

#tol=参与分解反应的最大次数，check_interval=每隔check_interval次检查反应成功率,max_iter为最大迭代次数，最多进行这么多次合成分解反应
def container(data_scaled,cluster,graph,adjmatrix,A_nomutual,G_nomutual,DG,tol,check_interval,max_iter,labels_true):
    print(len(data_scaled))
    sysrecord=defaultdict(int)
    sucessCount=0
    for i in range(max_iter):
        # print("第"+str(i)+"次迭代")
        cluster,sysrecord,isSucessSysthesis=synthesis(data_scaled,DG,G_nomutual,cluster,A_nomutual,sysrecord)
        cluster,graph,adjmatrix=updateCluster(cluster,data_scaled,A_nomutual)
        sucessCount+=isSucessSysthesis
        if i%check_interval==0 and i>0:
            # print("成功次数:"+str(sucessCount))
            if sucessCount<check_interval/5:
                cluster_temp=[]
                cluster_finished=[]
                adjmatrix_temp=[]
                adjmatrix_finished=[]
                for j in range(len(cluster)):
                    if sysrecord[frozenset(cluster[j].point)]<tol:
                        cluster_temp.append(cluster[j])
                        adjmatrix_temp.append(adjmatrix[j])
                    else:
                        cluster_finished.append(cluster[j])
                        adjmatrix_finished.append(cluster[j])
                if len(cluster_temp)==0:
                    break
                cluster=decomPose(data_scaled,cluster_temp,adjmatrix_temp)
                for i in range(len(cluster_finished)):
                    cluster.append(cluster_finished[i])
                cluster,graph,adjmatrix=updateCluster(cluster,data_scaled,A_nomutual)
                # print(len(cluster))
            sucessCount=0
        labels_pred=np.zeros(len(data_scaled))
        for i in range(len(cluster)):
            for element in cluster[i].point:
                labels_pred[element]=i
            print(cluster[i].count)
        ari_scores=adjusted_rand_score(labels_true,labels_pred)
        nmi_scores=normalized_mutual_info_score(labels_true,labels_pred)
        fmi_scores=fowlkes_mallows_score(labels_true,labels_pred)
        # print(ari_scores)
        # print(nmi_scores)
        # print(fmi_scores)
    if(len(cluster)==1):
        cluster=decomPose(data_scaled,cluster,adjmatrix)
    for i in range(len(cluster)):
        for element in cluster[i].point:
            labels_pred[element]=i
    return cluster,graph,adjmatrix,labels_pred
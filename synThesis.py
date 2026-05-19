import numpy as np
import networkx as nx
from rouletteWheelSelection import * 
from matrixTool import *
from Cluster import *
from sklearn.neighbors import KDTree

def getKE(data,A, res):
    KE = []
    selected = []
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            points_i = res[i].point
            points_j = res[j].point
            submatrix = A[np.ix_(points_i, points_j)]
            temp = np.sum(submatrix)
            if temp==0:
                u,v=findClostetPointPair(data,res[i],res[j])
                temp=np.exp(-euclidean_distance(data[u],data[v])**2/2)
            KE.append(temp)
            selected.append((i, j))
    return KE, selected


def getEa(DG,cluster):#修改了这里，将i改成了point[i],修改了G为DG
    Ea=[]
    for i in range(cluster.getCount()):
        temp=0
        for j in range(DG.number_of_nodes()):
            if DG.has_edge(cluster.point[i],j):
                temp+=DG[cluster.point[i]][j]['weight']
        Ea.append(temp)
    return Ea

def findByBFS(DG, start, cluster, Ea):#start为真正序号
    startEa=cluster.point.index(start)#转化为在point中的序号
    Layer = 0
    frontPoint = 0
    endPoint = 1  
    atomList = []
    atomList.append((start, Layer))
    
    layer_info = []  
    
    visited = np.zeros(DG.number_of_nodes())
    visited[start] = 1

    current_layer_max = Ea[startEa]
    current_layer_node = start
    current_layer_nodes = [] 
    
    while frontPoint < endPoint:
        current_atom, current_layer = atomList[frontPoint]
        current_atomEa = cluster.point.index(current_atom)
        frontPoint += 1
        
        if current_layer > Layer:
            layer_info.append((Layer, current_layer_max, current_layer_node))
            Layer = current_layer
            current_layer_max = Ea[current_atomEa]
            current_layer_node = current_atom
            current_layer_nodes = [current_atom]
        else:
            if Ea[current_atomEa] > current_layer_max:
                current_layer_max = Ea[current_atomEa]
                current_layer_node = current_atom
            current_layer_nodes.append(current_atom)
        
        for neighbor in DG.neighbors(current_atom):
            if neighbor in cluster.point and visited[neighbor] == 0:
                atomList.append((neighbor, current_layer + 1))
                visited[neighbor] = 1
                endPoint += 1
    
    if layer_info:
        if len(layer_info) == 1:
            centerAtom = layer_info[0][2]
        else:
            centerAtom = layer_info[-1][2]
            for i in range(len(layer_info)-1):
                if layer_info[i][1] > layer_info[i+1][1]:
                    centerAtom = layer_info[i][2]
                    return centerAtom
    else:
        centerAtom = start
    return centerAtom


def findCenterAtom(DG,data,cluster_1,cluster_2,A,Ea_a,Ea_b):
    u,v=findClostetPointPair(data,cluster_1,cluster_2)
    point_1=findByBFS(DG,u,cluster_1,Ea_a)
    point_2=findByBFS(DG,v,cluster_2,Ea_b)
    return point_1,point_2


def findClostetPointPair(data,cluster_1,cluster_2):#这里将返回的element_i,element_j改为u,v,disMin=temp之前没有
    points_1 = np.array(cluster_1.point)
    points_2 = np.array(cluster_2.point)

    if len(points_1) <= len(points_2):
        tree = KDTree(data[points_2])
        distance, indices = tree.query(data[points_1], k=1)
        local_index = int(np.argmin(distance[:, 0]))
        return int(points_1[local_index]), int(points_2[indices[local_index, 0]])

    tree = KDTree(data[points_1])
    distance, indices = tree.query(data[points_2], k=1)
    local_index = int(np.argmin(distance[:, 0]))
    return int(points_1[indices[local_index, 0]]), int(points_2[local_index])

def getArroundAtom(data,cluster,centerAtom,r):#判断条件<=r在括号之内
    Atom=[]
    for element in cluster.point:
        if euclidean_distance(data[centerAtom],data[element])<=r:
            Atom.append(element)
    return len(Atom)

def getBoundAtom(data,cluster_1,cluster_2,pointA,pointB,r):
    Atom=[]
    middlePoint=(data[pointA]+data[pointB])/2
    for element in cluster_1.point:
        if euclidean_distance(middlePoint,data[element])<=r:
            Atom.append(element)
    for element in cluster_2.point:
        if euclidean_distance(middlePoint,data[element])<=r:
            Atom.append(element)
    return len(Atom)+1

def conbineCluster(cluster_1,cluster_2):
    cluster_new=Cluster()
    for element in cluster_1.point:
        cluster_new.addPoint(element)
    for element in cluster_2.point:
        cluster_new.addPoint(element)
    return cluster_new

def isMeetCondition(Nb1,Nb2,Nt,PEmol_1,PEmol_2,PEmol_new):
    print('Nb1='+str(Nb1)+',Nb2='+str(Nb2)+',Nbt='+str(Nt)+',PE1='+str(PEmol_1)+',PE2='+str(PEmol_2)+',PE_new='+str(PEmol_new))
    RA_1=PEmol_new<PEmol_1 + PEmol_2
    RA_2=PEmol_1+PEmol_2>PEmol_new/2
    RB_1=Nt>=np.maximum(np.floor(min(Nb1, Nb2) * np.random.uniform(0.8, 1)), 1)
    # RB_1=Nt>=0.8*min(Nb1,Nb2)
    RB_2=(max(Nb1,Nb2)/min(Nb1,Nb2))<=1.5
    RB_3=Nt>((min(Nb1,Nb2))*2/3)
    RB_4=(max(Nb1,Nb2)/min(Nb1,Nb2))<5
    print(str(RA_1)+str(RA_2)+str(RB_1)+str(RB_2)+str(RB_3)+str(RB_4))
    if RA_1 or (RB_1 and RB_2):
        if not RA_1:
            if not RA_2:
                return False
        if not RB_1:
            if not RB_3:
                return False
        if not RB_2:
            if not RB_4:
                return False
    else:
        return False
    return True

def resetCluster(res,clusterPair,cluser_new):
    index_1=clusterPair[0]
    index_2=clusterPair[1]
    res_new=[]
    for i in range(len(res)):
        if i==index_1 or i==index_2:
            continue
        res_new.append(res[i])
    res_new.append(cluser_new)
    return res_new


def synthesis(data,DG,G,res,adjmatrix,sysrecord):#adjmatrix是A_nomutual
    if len(res)==1:
        return res,sysrecord,0
    KE,selected=getKE(data,adjmatrix,res)
    index=randWheelSelection(KE)
    # for i in range(len(res)):
    #      print(res[i].count)
    #      print("PE="+str(1/res[i].SecondSmallestEigenvalue()))
    print(selected[index])
    cluster_1=res[selected[index][0]]
    cluster_2=res[selected[index][1]]
    Ea_a=getEa(DG,cluster_1)
    Ea_b=getEa(DG,cluster_2)
    centers_a,centers_b=findCenterAtom(DG,data,cluster_1,cluster_2,adjmatrix,Ea_a,Ea_b)
    #centers_aEa=cluster_1.point.index(centers_a)
    #centers_bEa=cluster_2.point.index(centers_b)
    clostetPointPairA,clostetPointPairB=findClostetPointPair(data,cluster_1,cluster_2)
    r=max(euclidean_distance(data[clostetPointPairA],data[centers_a]),euclidean_distance(data[clostetPointPairB],data[centers_b]))
    print("center_a="+str(centers_a)+" clostet_a="+str(clostetPointPairA)+" center_b="+str(centers_b)+' clostet_b='+str(clostetPointPairB)+" r="+str(r))
    cluster_new=conbineCluster(cluster_1,cluster_2)
    cluster_new.initialClutser(data,adjmatrix)
    Nb1=getArroundAtom(data,cluster_1,centers_a,r)
    Nb2=getArroundAtom(data,cluster_2,centers_b,r)
    Nt=getBoundAtom(data,cluster_1,cluster_2,clostetPointPairA,clostetPointPairB,r)
    PEmol_1=cluster_1.PE
    PEmol_2=cluster_2.PE
    PEmol_new=cluster_new.PE
    if isMeetCondition(Nb1,Nb2,Nt,PEmol_1,PEmol_2,PEmol_new):
        sysrecord[frozenset(cluster_new.point)] += 1
        res_new=resetCluster(res,selected[index],cluster_new)
        isSucess=1
        #print("Sucess")
    else:
        res_new=res
        isSucess=0
        #print("noSucess")
    return res_new,sysrecord,isSucess

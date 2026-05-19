import numpy as np
import networkx as nx
import random
from Cluster import *


def conbineCluster(cluster_1,cluster_2):
    cluster_new=Cluster()
    for element in cluster_1.point:
        cluster_new.addPoint(element)
    for element in cluster_2.point:
        cluster_new.addPoint(element)
    return cluster_new

def randWheelSelection(values):
    F=0
    index=0
    qi=[]
    for i in range(len(values)):
        F+=values[i]
    pi=[element/F for element in values]
    for i in range(len(values)):
        temp=0
        for j in range(i+1):
            temp+=pi[j]
        qi.append(temp)
    rand_count=random.uniform(0.0,1.0)
    for i in range(len(values)):
        if qi[i]>rand_count:
            index=i
            break
    return index

def reassignCluster(res, index, DG):

    # 获取要重新分配的小聚类
    small_cluster = res[index]
    
    # 计算当前总点数（用于验证）
    total_points_before = sum(cluster.count for cluster in res)
    
    # 投票机制：找到小聚类应该合并到哪个相邻聚类
    vote = np.zeros(len(res))
    
    # 遍历小聚类中的每个点
    for point in small_cluster.point:
        # 查找该点的所有邻居
        for neighbor in DG.neighbors(point):
            # 检查邻居属于哪个聚类
            for j in range(len(res)):
                if neighbor in res[j].point:
                    vote[j] += 1
                    break  # 一个邻居只属于一个聚类
    
    # 排除自身（小聚类）
    vote[index] = 0
    
    # 找到得票最多的目标聚类
    if np.max(vote) > 0:  # 确保有有效的目标聚类
        target_index = np.argmax(vote)
        target_cluster = res[target_index]
        
        # 创建新的聚类列表
        res_new = []
        
        # 合并小聚类到目标聚类
        merged_cluster = Cluster()
        # 合并点集（确保没有重复）
        merged_points = list(set(target_cluster.point + small_cluster.point))
        merged_cluster.point = merged_points
        merged_cluster.count = len(merged_points)
        
        # 构建新的聚类列表
        for i in range(len(res)):
            if i == index:  # 跳过小聚类
                continue
            elif i == target_index:  # 替换目标聚类为合并后的聚类
                res_new.append(merged_cluster)
            else:  # 保留其他聚类不变
                res_new.append(res[i])
        
        return res_new
    else:
        return res


import numpy as np
import networkx as nx
from matrixTool import *

class Cluster:    
    def __init__(self):
        self.adjmatrix = []
        self.point = []
        self.count = 0
        self.G = nx.Graph()
        self.lysm = []
        self.second_eigval = None
        self.second_eigvec = None
        self.PE = None

    def initialClutser(self, data, A):
        self.knnGraph(data, A)
        self.adjMatrix()
        self.getNormalizedLaplacian()
        self.SecondSmallestEigenvalue()
        self.calculatePE()

    def addPoint(self, x):
        self.count += 1
        self.point.append(x)
    
    def getPoint(self):
        return self.point
    
    def getCount(self):
        return self.count
    
    def knnGraph(self, data, A):
        """修正：确保节点和边使用一致的标识"""
        self.G = nx.Graph()
        
        # 添加节点，使用局部索引
        for i in range(self.count):
            self.G.add_node(i, pos=data[self.point[i]], original_index=self.point[i])
        
        # 添加边，使用局部索引
        for i in range(self.count):
            for j in range(i+1, self.count):  # 修正：避免重复添加
                original_i, original_j = self.point[i], self.point[j]
                if A[original_i][original_j] != 0:
                    self.G.add_edge(i, j, weight=A[original_i][original_j])
        return self.G
    
    def adjMatrix(self):
        """修正：确保邻接矩阵正确反映图结构"""
        self.adjmatrix = np.zeros((self.count, self.count))
        for i, j in self.G.edges():
            weight = self.G[i][j]['weight']
            self.adjmatrix[i][j] = weight
            self.adjmatrix[j][i] = weight
        return self.adjmatrix
    
    def getNormalizedLaplacian(self):
        self.lysm = normalizedLaplacian(self.adjmatrix)
        return self.lysm
    
    def SecondSmallestEigenvalue(self):
        self.second_eigval = getSecondSmallestEigenvalue(self.lysm)
        self.second_eigvec = getSecondSmallestEigenvector(self.lysm)
        return self.second_eigval
    
    def calculatePE(self):
        """与原算法一致的PE计算方法"""
        # if self.count <= 4:
        #     self.PE = -np.inf
        #     return
        
        if self.second_eigvec is None:
            self.SecondSmallestEigenvalue()
        
        # 根据特征向量划分节点
        positive_nodes = []
        negative_nodes = []
        
        for i in range(len(self.second_eigvec)):
            if self.second_eigvec[i] > 0:
                positive_nodes.append(i)
            else:
                negative_nodes.append(i)
        
        # 处理特殊情况
        if not positive_nodes or not negative_nodes:
            self.PE = 0
            return
        
        # 计算度矩阵D
        degree = np.sum(self.adjmatrix, axis=1)
        D = np.diag(degree)
        
        # 计算体积（与原算法一致）
        vol_A = np.sum(D[positive_nodes, :][:, positive_nodes])
        vol_B = np.sum(D[negative_nodes, :][:, negative_nodes])
        
        # 使用原算法的PE计算公式
        Cut = (np.abs(self.second_eigval) / (1 / vol_A + 1 / vol_B))
        self.PE = 1 / np.abs(Cut)
    
    def getPE(self):
        if self.PE is None:
            self.calculatePE()
        return self.PE
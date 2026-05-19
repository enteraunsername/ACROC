import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def euclidean_distance(a, b):#计算a,b两点之间的欧式距离
    return np.linalg.norm(a-b)

def getDegreeVector(adj_matrix, epsilon=1e-8):
    degree = np.sum(adj_matrix, axis=1)
    degree[degree == 0] = epsilon
    return degree

def normalizedLaplacian(adj_matrix, epsilon=1e-8):
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    degree = getDegreeVector(adj_matrix, epsilon)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degree))
    n = adj_matrix.shape[0]
    L_norm = np.eye(n) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
    L_norm = (L_norm + L_norm.T) / 2
    L_norm += np.eye(n) * epsilon
    return L_norm

def _second_smallest_eigenpair(matrix):
    n = matrix.shape[0]
    if n == 0:
        return 0.0, np.array([])
    if n == 1:
        return float(matrix[0][0]), np.ones(1)

    try:
        if n > 256:
            sparse_matrix = sparse.csr_matrix(matrix)
            eigenvalues, eigenvectors = eigsh(
                sparse_matrix,
                k=2,
                which="SM",
                tol=1e-5,
                maxiter=max(1000, n * 20),
            )
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

    sorted_indices = np.argsort(eigenvalues)
    second_index = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
    return float(np.real(eigenvalues[second_index])), np.real(eigenvectors[:, second_index])

def getSecondSmallestEigenpair(matrix):
    return _second_smallest_eigenpair(matrix)

def getSecondSmallestEigenvalue(matrix):
    if matrix.shape[0] > 256:
        return _second_smallest_eigenpair(matrix)[0]
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues = np.real(eigenvalues)
    sorted_eigenvalues = np.sort(eigenvalues)
    if len(sorted_eigenvalues) > 1:
        return sorted_eigenvalues[1]
    else:
        return sorted_eigenvalues[0]  

def getSecondSmallestEigenvector(matrix):
    if matrix.shape[0] > 256:
        return _second_smallest_eigenpair(matrix)[1]
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
    
    # 检查特征向量矩阵的大小
    n = eigenvectors.shape[1] if eigenvectors.shape else 0
    
    # 如果特征向量数量不足2个，返回第一个特征向量（或全1向量）
    if n < 2:
        if n == 1:
            return eigenvectors[:, 0]
        else:
            # 如果没有特征向量，返回一个全1向量
            return np.ones(matrix.shape[0]) if matrix.shape[0] > 0 else np.array([])
    
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 确保索引1存在
    if sorted_eigenvectors.shape[1] > 1:
        return sorted_eigenvectors[:, 1]
    else:
        return sorted_eigenvectors[:, 0]

def getPE(eigenvalue, epsilon=1e-8):
    if abs(eigenvalue) < epsilon:
        return 1.0 / epsilon
    return 1.0 / eigenvalue

def getPointVector(adj_matrix, eigenvector):
    degree_vector = getDegreeVector(adj_matrix)
    return degree_vector**(-0.5) * eigenvector

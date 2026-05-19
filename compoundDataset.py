import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class CompoundDatasetRNG:
    """
    Compound数据集随机数生成器
    生成具有复杂形状和层次结构的Compound数据集
    """
    
    def __init__(self, seed=None, scale_factor=1.0):
        """
        初始化Compound数据集随机数生成器
        
        Parameters:
        - seed: 随机种子，用于可重复性
        - scale_factor: 缩放因子，用于调整数据范围大小
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.scale_factor = scale_factor
    
    def get_rng(self):
        """获取随机数生成器实例"""
        return self.rng
    
    def reset_rng(self, seed=None):
        """重置随机数生成器"""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        return self.rng
    
    def set_scale_factor(self, scale_factor):
        """设置缩放因子，调整数据范围大小"""
        self.scale_factor = scale_factor
    
    def generate_compound(self, n_samples=500, noise=0.05, return_centers=False):
        """
        生成Compound数据集
        
        Parameters:
        - n_samples: 总样本数
        - noise: 噪声水平
        - return_centers: 是否返回簇中心
        
        Returns:
        - X: 特征数组 (n_samples, 2)
        - y: 标签数组 (n_samples,)
        - centers: 如果return_centers为True，返回簇中心
        """
        # 应用缩放因子调整数据范围
        sf = self.scale_factor
        
        # Compound数据集通常包含多个不同形状和大小的簇
        # 我们将创建几个不同形状的簇来模拟Compound数据集
        
        # 簇1: 大圆形簇 (中心区域)
        n1 = int(n_samples * 0.3)
        center1 = [0, 0]
        radius1 = 2.0 * sf
        theta1 = self.rng.uniform(0, 2*np.pi, n1)
        r1 = self.rng.uniform(0, radius1, n1)
        x1 = r1 * np.cos(theta1) + center1[0]
        y1 = r1 * np.sin(theta1) + center1[1]
        
        # 簇2: 小圆形簇 (右上角)
        n2 = int(n_samples * 0.15)
        center2 = [4 * sf, 4 * sf]
        radius2 = 0.8 * sf
        theta2 = self.rng.uniform(0, 2*np.pi, n2)
        r2 = self.rng.uniform(0, radius2, n2)
        x2 = r2 * np.cos(theta2) + center2[0]
        y2 = r2 * np.sin(theta2) + center2[1]
        
        # 簇3: 小圆形簇 (右下角)
        n3 = int(n_samples * 0.15)
        center3 = [4 * sf, -4 * sf]
        radius3 = 0.8 * sf
        theta3 = self.rng.uniform(0, 2*np.pi, n3)
        r3 = self.rng.uniform(0, radius3, n3)
        x3 = r3 * np.cos(theta3) + center3[0]
        y3 = r3 * np.sin(theta3) + center3[1]
        
        # 簇4: 半月形簇 (左侧)
        n4 = int(n_samples * 0.2)
        center4 = [-3 * sf, 0]
        theta4 = self.rng.uniform(np.pi/2, 3*np.pi/2, n4)
        r4 = self.rng.uniform(1.0 * sf, 2.0 * sf, n4)
        x4 = r4 * np.cos(theta4) + center4[0]
        y4 = r4 * np.sin(theta4) + center4[1]
        
        # 簇5: 椭圆形簇 (左上角)
        n5 = int(n_samples * 0.2)
        center5 = [-2 * sf, 3 * sf]
        theta5 = self.rng.uniform(0, 2*np.pi, n5)
        # 椭圆形: 不同方向上的半径不同
        r5_x = self.rng.uniform(0, 1.5 * sf, n5)
        r5_y = self.rng.uniform(0, 0.8 * sf, n5)
        x5 = r5_x * np.cos(theta5) + center5[0]
        y5 = r5_y * np.sin(theta5) + center5[1]
        
        # 合并所有簇
        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2]),
            np.column_stack([x3, y3]),
            np.column_stack([x4, y4]),
            np.column_stack([x5, y5])
        ])
        
        # 创建标签
        y = np.hstack([
            np.zeros(n1),   # 簇1
            np.ones(n2),    # 簇2
            np.full(n3, 2), # 簇3
            np.full(n4, 3), # 簇4
            np.full(n5, 4)  # 簇5
        ])
        
        # 添加噪声
        X += self.rng.normal(0, noise * sf, X.shape)
        
        # 计算簇中心
        centers = np.array([center1, center2, center3, center4, center5])
        
        if return_centers:
            return X, y, centers
        return X, y
    
    def generate_compound_variant(self, n_samples=600, noise=0.05, return_centers=False):
        """
        生成另一种变体的Compound数据集
        
        Parameters:
        - n_samples: 总样本数
        - noise: 噪声水平
        - return_centers: 是否返回簇中心
        
        Returns:
        - X: 特征数组 (n_samples, 2)
        - y: 标签数组 (n_samples,)
        - centers: 如果return_centers为True，返回簇中心
        """
        # 应用缩放因子
        sf = self.scale_factor
        
        # 这个变体包含更多不同形状和大小的簇
        
        # 簇1: 大圆形簇 (中心)
        n1 = int(n_samples * 0.25)
        center1 = [0, 0]
        radius1 = 2.5 * sf
        theta1 = self.rng.uniform(0, 2*np.pi, n1)
        r1 = self.rng.uniform(0, radius1, n1)
        x1 = r1 * np.cos(theta1) + center1[0]
        y1 = r1 * np.sin(theta1) + center1[1]
        
        # 簇2: 小圆形簇 (右上)
        n2 = int(n_samples * 0.1)
        center2 = [5 * sf, 5 * sf]
        radius2 = 0.7 * sf
        theta2 = self.rng.uniform(0, 2*np.pi, n2)
        r2 = self.rng.uniform(0, radius2, n2)
        x2 = r2 * np.cos(theta2) + center2[0]
        y2 = r2 * np.sin(theta2) + center2[1]
        
        # 簇3: 小圆形簇 (右下)
        n3 = int(n_samples * 0.1)
        center3 = [5 * sf, -5 * sf]
        radius3 = 0.7 * sf
        theta3 = self.rng.uniform(0, 2*np.pi, n3)
        r3 = self.rng.uniform(0, radius3, n3)
        x3 = r3 * np.cos(theta3) + center3[0]
        y3 = r3 * np.sin(theta3) + center3[1]
        
        # 簇4: 半月形簇 (左侧)
        n4 = int(n_samples * 0.15)
        center4 = [-4 * sf, 0]
        theta4 = self.rng.uniform(np.pi/3, 5*np.pi/3, n4)
        r4 = self.rng.uniform(1.5 * sf, 2.5 * sf, n4)
        x4 = r4 * np.cos(theta4) + center4[0]
        y4 = r4 * np.sin(theta4) + center4[1]
        
        # 簇5: 椭圆形簇 (左上)
        n5 = int(n_samples * 0.15)
        center5 = [-3 * sf, 4 * sf]
        theta5 = self.rng.uniform(0, 2*np.pi, n5)
        r5_x = self.rng.uniform(0, 1.2 * sf, n5)
        r5_y = self.rng.uniform(0, 0.6 * sf, n5)
        x5 = r5_x * np.cos(theta5) + center5[0]
        y5 = r5_y * np.sin(theta5) + center5[1]
        
        # 簇6: 弧形簇 (左下)
        n6 = int(n_samples * 0.15)
        center6 = [-3 * sf, -4 * sf]
        theta6 = self.rng.uniform(7*np.pi/6, 11*np.pi/6, n6)
        r6 = self.rng.uniform(1.0 * sf, 2.0 * sf, n6)
        x6 = r6 * np.cos(theta6) + center6[0]
        y6 = r6 * np.sin(theta6) + center6[1]
        
        # 簇7: 小圆形簇 (中上)
        n7 = int(n_samples * 0.1)
        center7 = [0, 3 * sf]
        radius7 = 0.5 * sf
        theta7 = self.rng.uniform(0, 2*np.pi, n7)
        r7 = self.rng.uniform(0, radius7, n7)
        x7 = r7 * np.cos(theta7) + center7[0]
        y7 = r7 * np.sin(theta7) + center7[1]
        
        # 合并所有簇
        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2]),
            np.column_stack([x3, y3]),
            np.column_stack([x4, y4]),
            np.column_stack([x5, y5]),
            np.column_stack([x6, y6]),
            np.column_stack([x7, y7])
        ])
        
        # 创建标签
        y = np.hstack([
            np.zeros(n1),   # 簇1
            np.ones(n2),    # 簇2
            np.full(n3, 2), # 簇3
            np.full(n4, 3), # 簇4
            np.full(n5, 4), # 簇5
            np.full(n6, 5), # 簇6
            np.full(n7, 6)  # 簇7
        ])
        
        # 添加噪声
        X += self.rng.normal(0, noise * sf, X.shape)
        
        # 计算簇中心
        centers = np.array([center1, center2, center3, center4, center5, center6, center7])
        
        if return_centers:
            return X, y, centers
        return X, y
    
    def generate_large_scale_compound(self, n_samples=1000, noise=0.05, return_centers=False):
        """
        生成大规模Compound数据集（数值范围更大）
        
        Parameters:
        - n_samples: 总样本数
        - noise: 噪声水平
        - return_centers: 是否返回簇中心
        
        Returns:
        - X: 特征数组 (n_samples, 2)
        - y: 标签数组 (n_samples,)
        - centers: 如果return_centers为True，返回簇中心
        """
        # 使用更大的缩放因子
        original_scale = self.scale_factor
        self.scale_factor = 5.0  # 临时使用更大的缩放因子
        
        # 生成数据
        X, y = self.generate_compound(n_samples=n_samples, noise=noise)
        
        # 恢复原始缩放因子
        self.scale_factor = original_scale
        
        if return_centers:
            # 计算中心点
            centers = []
            for label in np.unique(y):
                cluster_points = X[y == label]
                centers.append(np.mean(cluster_points, axis=0))
            return X, y, np.array(centers)
        
        return X, y
    
    def visualize_compound(self, n_samples=500, variant='standard', figsize=(12, 8)):
        """
        可视化Compound数据集
        
        Parameters:
        - n_samples: 样本数量
        - variant: 数据集变体 ('standard' 或 'variant')
        - figsize: 图形大小
        """
        if variant == 'standard':
            X, y = self.generate_compound(n_samples=n_samples)
            title = 'Standard Compound Dataset'
        elif variant == 'variant':
            X, y = self.generate_compound_variant(n_samples=n_samples)
            title = 'Compound Dataset Variant'
        elif variant == 'large':
            X, y = self.generate_large_scale_compound(n_samples=n_samples)
            title = 'Large Scale Compound Dataset'
        else:
            raise ValueError("variant 必须是 'standard', 'variant' 或 'large'")
        
        plt.figure(figsize=figsize)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)
        plt.title(f'{title} (n={n_samples}, scale={self.scale_factor})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
        
        return X, y
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'name': 'Compound Dataset',
            'description': '包含多个不同形状、大小和密度的簇的复杂数据集',
            'variants': {
                'standard': '标准Compound数据集，包含5个不同形状的簇',
                'variant': '变体Compound数据集，包含7个不同形状的簇',
                'large': '大规模Compound数据集，数值范围更大'
            },
            'typical_use': '用于测试聚类算法对复杂形状和不同密度簇的识别能力',
            'scale_factor': self.scale_factor
        }
        return info

# 使用示例
if __name__ == "__main__":
    # 创建Compound数据集生成器，设置较大的缩放因子
    compound_gen = CompoundDatasetRNG(seed=500, scale_factor=10.0)
    
    # 获取数据集信息
    info = compound_gen.get_dataset_info()
    print(f"数据集名称: {info['name']}")
    print(f"描述: {info['description']}")
    print(f"典型用途: {info['typical_use']}")
    print(f"当前缩放因子: {info['scale_factor']}")
    
    # 生成标准Compound数据集（数值范围更大）
    X_standard, y_standard = compound_gen.generate_compound(n_samples=500)
    print(f"标准Compound数据集: {X_standard.shape[0]}个样本, {X_standard.shape[1]}个特征, {len(np.unique(y_standard))}个簇")
    print(f"数据范围: X∈[{X_standard[:,0].min():.2f}, {X_standard[:,0].max():.2f}], Y∈[{X_standard[:,1].min():.2f}, {X_standard[:,1].max():.2f}]")
    
    # 生成大规模Compound数据集
    X_large, y_large = compound_gen.generate_large_scale_compound(n_samples=1000)
    print(f"大规模Compound数据集: {X_large.shape[0]}个样本, {X_large.shape[1]}个特征, {len(np.unique(y_large))}个簇")
    print(f"数据范围: X∈[{X_large[:,0].min():.2f}, {X_large[:,0].max():.2f}], Y∈[{X_large[:,1].min():.2f}, {X_large[:,1].max():.2f}]")
    
    # 可视化数据集
    compound_gen.visualize_compound(n_samples=500, variant='standard')
    compound_gen.visualize_compound(n_samples=1000, variant='large')
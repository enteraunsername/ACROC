import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class JainDatasetRNG:
    """
    Jain数据集随机数生成器
    生成经典的Jain数据集（两个不规则形状的簇）
    """
    
    def __init__(self, seed=None, scale_factor=1.0):
        """
        初始化Jain数据集随机数生成器
        
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
    
    def generate_jain(self, n_samples=500, noise=0.05, return_centers=False):
        """
        生成经典Jain数据集
        
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
        
        # Jain数据集包含两个不规则形状的簇
        
        # 簇1: 弯曲的弧形簇
        n1 = n_samples // 2
        t1 = self.rng.uniform(0, 1, n1)
        x1 = 4 * t1 + 0.5 * np.sin(4 * np.pi * t1) + 8
        y1 = 0.5 * np.sin(2 * np.pi * t1) + 8
        
        # 簇2: 椭圆形簇
        n2 = n_samples - n1
        theta2 = self.rng.uniform(0, 2 * np.pi, n2)
        r2 = self.rng.uniform(0, 1.5, n2)
        x2 = 2 * r2 * np.cos(theta2) + 2
        y2 = 0.8 * r2 * np.sin(theta2) + 2
        
        # 合并数据
        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.hstack([np.zeros(n1), np.ones(n2)])
        
        # 应用缩放因子
        X = X * sf
        
        # 添加噪声
        X += self.rng.normal(0, noise * sf, X.shape)
        
        # 计算簇中心
        centers = np.array([
            [np.mean(x1) * sf, np.mean(y1) * sf],
            [np.mean(x2) * sf, np.mean(y2) * sf]
        ])
        
        if return_centers:
            return X, y, centers
        return X, y
    
    def generate_jain_variant(self, n_samples=500, noise=0.05, return_centers=False):
        """
        生成变体Jain数据集
        
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
        
        # 变体Jain数据集 - 调整簇的形状和位置
        
        # 簇1: 更弯曲的弧形簇
        n1 = n_samples // 2
        t1 = self.rng.uniform(0, 1, n1)
        x1 = 3.5 * t1 + 0.7 * np.sin(5 * np.pi * t1) + 7
        y1 = 0.6 * np.sin(3 * np.pi * t1) + 7
        
        # 簇2: 更扁平的椭圆形簇
        n2 = n_samples - n1
        theta2 = self.rng.uniform(0, 2 * np.pi, n2)
        r2 = self.rng.uniform(0, 1.2, n2)
        x2 = 1.8 * r2 * np.cos(theta2) + 1.5
        y2 = 0.5 * r2 * np.sin(theta2) + 1.5
        
        # 合并数据
        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.hstack([np.zeros(n1), np.ones(n2)])
        
        # 应用缩放因子
        X = X * sf
        
        # 添加噪声
        X += self.rng.normal(0, noise * sf, X.shape)
        
        # 计算簇中心
        centers = np.array([
            [np.mean(x1) * sf, np.mean(y1) * sf],
            [np.mean(x2) * sf, np.mean(y2) * sf]
        ])
        
        if return_centers:
            return X, y, centers
        return X, y
    
    def generate_large_scale_jain(self, n_samples=1000, noise=0.05, return_centers=False):
        """
        生成大规模Jain数据集（数值范围更大）
        
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
        X, y = self.generate_jain(n_samples=n_samples, noise=noise)
        
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
    
    def visualize_jain(self, n_samples=500, variant='standard', figsize=(12, 8)):
        """
        可视化Jain数据集
        
        Parameters:
        - n_samples: 样本数量
        - variant: 数据集变体 ('standard' 或 'variant')
        - figsize: 图形大小
        """
        if variant == 'standard':
            X, y = self.generate_jain(n_samples=n_samples)
            title = 'Standard Jain Dataset'
        elif variant == 'variant':
            X, y = self.generate_jain_variant(n_samples=n_samples)
            title = 'Jain Dataset Variant'
        elif variant == 'large':
            X, y = self.generate_large_scale_jain(n_samples=n_samples)
            title = 'Large Scale Jain Dataset'
        else:
            raise ValueError("variant 必须是 'standard', 'variant' 或 'large'")
        
        plt.figure(figsize=figsize)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', alpha=0.7, s=30)
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
            'name': 'Jain Dataset',
            'description': '经典的Jain数据集，包含两个不规则形状的簇',
            'variants': {
                'standard': '标准Jain数据集，两个不规则形状簇',
                'variant': '变体Jain数据集，簇形状略有不同',
                'large': '大规模Jain数据集，数值范围更大'
            },
            'typical_use': '用于测试聚类算法对复杂形状和不规则簇的识别能力',
            'scale_factor': self.scale_factor,
            'cluster_count': 2
        }
        return info

# 使用示例
if __name__ == "__main__":
    # 创建Jain数据集生成器
    jain_gen = JainDatasetRNG(seed=42, scale_factor=1.0)
    
    # 获取数据集信息
    info = jain_gen.get_dataset_info()
    print(f"数据集名称: {info['name']}")
    print(f"描述: {info['description']}")
    print(f"典型用途: {info['typical_use']}")
    print(f"簇数量: {info['cluster_count']}")
    print(f"当前缩放因子: {info['scale_factor']}")
    
    # 生成标准Jain数据集
    X_standard, y_standard = jain_gen.generate_jain(n_samples=500)
    print(f"标准Jain数据集: {X_standard.shape[0]}个样本, {X_standard.shape[1]}个特征, {len(np.unique(y_standard))}个簇")
    print(f"数据范围: X∈[{X_standard[:,0].min():.2f}, {X_standard[:,0].max():.2f}], Y∈[{X_standard[:,1].min():.2f}, {X_standard[:,1].max():.2f}]")
    
    # 生成变体Jain数据集
    X_variant, y_variant = jain_gen.generate_jain_variant(n_samples=500)
    print(f"变体Jain数据集: {X_variant.shape[0]}个样本, {X_variant.shape[1]}个特征, {len(np.unique(y_variant))}个簇")
    print(f"数据范围: X∈[{X_variant[:,0].min():.2f}, {X_variant[:,0].max():.2f}], Y∈[{X_variant[:,1].min():.2f}, {X_variant[:,1].max():.2f}]")
    
    # 生成大规模Jain数据集
    jain_gen.set_scale_factor(3.0)
    X_large, y_large = jain_gen.generate_large_scale_jain(n_samples=1000)
    print(f"大规模Jain数据集: {X_large.shape[0]}个样本, {X_large.shape[1]}个特征, {len(np.unique(y_large))}个簇")
    print(f"数据范围: X∈[{X_large[:,0].min():.2f}, {X_large[:,0].max():.2f}], Y∈[{X_large[:,1].min():.2f}, {X_large[:,1].max():.2f}]")
    
    # 可视化数据集
    jain_gen.visualize_jain(n_samples=500, variant='standard')
    jain_gen.visualize_jain(n_samples=500, variant='variant')
    jain_gen.visualize_jain(n_samples=1000, variant='large')
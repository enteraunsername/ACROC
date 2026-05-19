import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class SpiralDatasetRNG:
    """
    螺旋(Spiral)数据集随机数生成器
    生成螺旋形状的数据集，包含多个螺旋形簇
    """
    
    def __init__(self, seed=None, scale_factor=1.0):
        """
        初始化螺旋数据集随机数生成器
        
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
    
    def generate_spiral(self, n_samples=600, noise=0.05, n_spirals=3, return_centers=False):
        """
        生成螺旋数据集
        
        Parameters:
        - n_samples: 总样本数
        - noise: 噪声水平
        - n_spirals: 螺旋数量
        - return_centers: 是否返回簇中心
        
        Returns:
        - X: 特征数组 (n_samples, 2)
        - y: 标签数组 (n_samples,)
        - centers: 如果return_centers为True，返回簇中心
        """
        # 应用缩放因子调整数据范围
        sf = self.scale_factor
        
        points_per_spiral = n_samples // n_spirals
        X_list = []
        y_list = []
        centers = []
        
        for i in range(n_spirals):
            # 每个螺旋的参数
            n = points_per_spiral
            # 使用平方根使点分布更均匀
            theta = np.sqrt(self.rng.uniform(0, 6.5*np.pi, n))
            
            # 螺旋公式
            r = 0.4 * theta * sf
            x = r * np.cos(theta + 2*np.pi*i/n_spirals)
            y = r * np.sin(theta + 2*np.pi*i/n_spirals)
            
            # 添加噪声
            x += self.rng.normal(0, noise * sf, n)
            y += self.rng.normal(0, noise * sf, n)
            
            X_list.append(np.column_stack([x, y]))
            y_list.append(np.full(n, i))
            
            # 计算每个螺旋的中心（近似中心）
            centers.append([np.mean(x), np.mean(y)])
        
        # 合并所有螺旋
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # 如果样本数不整除，添加额外样本到最后一个螺旋
        if len(X) < n_samples:
            extra_n = n_samples - len(X)
            i = n_spirals - 1  # 最后一个螺旋
            theta = np.sqrt(self.rng.uniform(0, 6.5*np.pi, extra_n))
            r = 0.4 * theta * sf
            x = r * np.cos(theta + 2*np.pi*i/n_spirals)
            y = r * np.sin(theta + 2*np.pi*i/n_spirals)
            x += self.rng.normal(0, noise * sf, extra_n)
            y += self.rng.normal(0, noise * sf, extra_n)
            
            X = np.vstack([X, np.column_stack([x, y])])
            y = np.hstack([y, np.full(extra_n, i)])
        
        if return_centers:
            return X, y, np.array(centers)
        return X, y
    
    def generate_spiral_variant(self, n_samples=600, noise=0.05, return_centers=False):
        """
        生成变体螺旋数据集
        
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
        
        # 变体螺旋数据集 - 调整螺旋的形状和参数
        
        # 簇1: 紧密螺旋
        n1 = n_samples // 3
        theta1 = np.sqrt(self.rng.uniform(0, 5*np.pi, n1))
        r1 = 0.3 * theta1 * sf
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)
        
        # 簇2: 宽松螺旋
        n2 = n_samples // 3
        theta2 = np.sqrt(self.rng.uniform(0, 8*np.pi, n2))
        r2 = 0.5 * theta2 * sf
        x2 = r2 * np.cos(theta2 + 2*np.pi/3)
        y2 = r2 * np.sin(theta2 + 2*np.pi/3)
        
        # 簇3: 反向螺旋
        n3 = n_samples - n1 - n2
        theta3 = np.sqrt(self.rng.uniform(0, 7*np.pi, n3))
        r3 = 0.4 * theta3 * sf
        x3 = r3 * np.cos(-theta3 + 4*np.pi/3)  # 负号表示反向旋转
        y3 = r3 * np.sin(-theta3 + 4*np.pi/3)
        
        # 合并所有簇
        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2]),
            np.column_stack([x3, y3])
        ])
        
        # 创建标签
        y = np.hstack([
            np.zeros(n1),   # 紧密螺旋
            np.ones(n2),    # 宽松螺旋
            np.full(n3, 2)  # 反向螺旋
        ])
        
        # 添加噪声
        X += self.rng.normal(0, noise * sf, X.shape)
        
        # 计算簇中心
        centers = np.array([
            [np.mean(x1), np.mean(y1)],
            [np.mean(x2), np.mean(y2)],
            [np.mean(x3), np.mean(y3)]
        ])
        
        if return_centers:
            return X, y, centers
        return X, y
    
    def generate_large_scale_spiral(self, n_samples=1000, noise=0.05, n_spirals=3, return_centers=False):
        """
        生成大规模螺旋数据集（数值范围更大）
        
        Parameters:
        - n_samples: 总样本数
        - noise: 噪声水平
        - n_spirals: 螺旋数量
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
        X, y = self.generate_spiral(n_samples=n_samples, noise=noise, n_spirals=n_spirals)
        
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
    
    def visualize_spiral(self, n_samples=600, variant='standard', figsize=(12, 10)):
        """
        可视化螺旋数据集
        
        Parameters:
        - n_samples: 样本数量
        - variant: 数据集变体 ('standard' 或 'variant')
        - figsize: 图形大小
        """
        if variant == 'standard':
            X, y = self.generate_spiral(n_samples=n_samples, n_spirals=3)
            title = 'Standard Spiral Dataset'
        elif variant == 'variant':
            X, y = self.generate_spiral_variant(n_samples=n_samples)
            title = 'Spiral Dataset Variant'
        elif variant == 'large':
            X, y = self.generate_large_scale_spiral(n_samples=n_samples, n_spirals=3)
            title = 'Large Scale Spiral Dataset'
        else:
            raise ValueError("variant 必须是 'standard', 'variant' 或 'large'")
        
        plt.figure(figsize=figsize)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', alpha=0.7, s=30)
        plt.title(f'{title} (n={n_samples}, scale={self.scale_factor})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 添加螺旋特征说明
        if variant == 'standard':
            label_names = {
                0: 'Spiral 1',
                1: 'Spiral 2', 
                2: 'Spiral 3'
            }
        elif variant == 'variant':
            label_names = {
                0: 'Tight Spiral',
                1: 'Loose Spiral',
                2: 'Reverse Spiral'
            }
        else:
            label_names = {i: f'Spiral {i+1}' for i in range(len(np.unique(y)))}
        
        # 在图上标注各个螺旋
        for label in np.unique(y):
            cluster_points = X[y == label]
            # 找到螺旋的外端点作为标注位置
            distances = np.sqrt(np.sum(cluster_points**2, axis=1))
            outer_point_idx = np.argmax(distances)
            outer_point = cluster_points[outer_point_idx]
            
            plt.text(outer_point[0], outer_point[1], label_names[label], 
                    fontsize=12, fontweight='bold', ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        plt.show()
        
        return X, y
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'name': 'Spiral Dataset',
            'description': '螺旋形状的数据集，包含多个螺旋形簇',
            'variants': {
                'standard': '标准螺旋数据集，包含3个螺旋形簇',
                'variant': '变体螺旋数据集，不同形状和旋转方向的螺旋',
                'large': '大规模螺旋数据集，数值范围更大'
            },
            'typical_use': '用于测试聚类算法对复杂非线性结构的识别能力',
            'scale_factor': self.scale_factor,
            'cluster_count': '可配置（通常为3）',
            'special_characteristics': '高度非线性的数据结构，对基于距离的聚类算法极具挑战性'
        }
        return info

# 使用示例
if __name__ == "__main__":
    # 创建螺旋数据集生成器
    spiral_gen = SpiralDatasetRNG(seed=42, scale_factor=1.0)
    
    # 获取数据集信息
    info = spiral_gen.get_dataset_info()
    print(f"数据集名称: {info['name']}")
    print(f"描述: {info['description']}")
    print(f"典型用途: {info['typical_use']}")
    print(f"簇数量: {info['cluster_count']}")
    print(f"特殊特征: {info['special_characteristics']}")
    print(f"当前缩放因子: {info['scale_factor']}")
    
    # 生成标准螺旋数据集
    X_standard, y_standard = spiral_gen.generate_spiral(n_samples=600, n_spirals=3)
    print(f"标准螺旋数据集: {X_standard.shape[0]}个样本, {X_standard.shape[1]}个特征, {len(np.unique(y_standard))}个簇")
    print(f"数据范围: X∈[{X_standard[:,0].min():.2f}, {X_standard[:,0].max():.2f}], Y∈[{X_standard[:,1].min():.2f}, {X_standard[:,1].max():.2f}]")
    
    # 生成变体螺旋数据集
    X_variant, y_variant = spiral_gen.generate_spiral_variant(n_samples=600)
    print(f"变体螺旋数据集: {X_variant.shape[0]}个样本, {X_variant.shape[1]}个特征, {len(np.unique(y_variant))}个簇")
    print(f"数据范围: X∈[{X_variant[:,0].min():.2f}, {X_variant[:,0].max():.2f}], Y∈[{X_variant[:,1].min():.2f}, {X_variant[:,1].max():.2f}]")
    
    # 生成大规模螺旋数据集
    spiral_gen.set_scale_factor(3.0)
    X_large, y_large = spiral_gen.generate_large_scale_spiral(n_samples=1000, n_spirals=3)
    print(f"大规模螺旋数据集: {X_large.shape[0]}个样本, {X_large.shape[1]}个特征, {len(np.unique(y_large))}个簇")
    print(f"数据范围: X∈[{X_large[:,0].min():.2f}, {X_large[:,0].max():.2f}], Y∈[{X_large[:,1].min():.2f}, {X_large[:,1].max():.2f}]")
    
    # 可视化数据集
    spiral_gen.visualize_spiral(n_samples=600, variant='standard')
    spiral_gen.visualize_spiral(n_samples=600, variant='variant')
    spiral_gen.visualize_spiral(n_samples=1000, variant='large')
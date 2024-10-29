'''
Author: huangjie huangjie20011001@163.com
Date: 2024-10-29 13:04:07
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def gen_adjmatrix(time_length: int = 6, bin_length: int = 4, direct: bool = False) -> np.ndarray:
    '''生成邻接矩阵'''
    num_nodes = time_length * bin_length
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        row, col = divmod(i, bin_length)
        # 右侧邻居
        if col < bin_length - 1:
            right_neighbor = i + 1
            adjacency_matrix[i, right_neighbor] = 1
            if not direct:
                adjacency_matrix[right_neighbor, i] = 1
        # 下方邻居
        if row < time_length - 1:
            bottom_neighbor = i + bin_length
            adjacency_matrix[i, bottom_neighbor] = 1
            if not direct:
                adjacency_matrix[bottom_neighbor, i] = 1
    return adjacency_matrix

def draw_grid_with_features(adjacency_matrix, time_length, bin_length, features, feature_index=0):
    '''绘制邻接矩阵'''
    G = nx.DiGraph() if np.any(adjacency_matrix != adjacency_matrix.T) else nx.Graph()
    num_nodes = time_length * bin_length
    G.add_nodes_from(range(num_nodes))

    # 添加边
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    # 设置布局
    pos = {i: (i % bin_length, time_length - 1 - i // bin_length) for i in range(num_nodes)}
    
    # 绘制图，并在节点上显示选定的特征值
    plt.figure(figsize=(8, 6))
    labels = {i: f'{i}\n{features[i, feature_index]:.2f}' for i in range(num_nodes)}  # 创建带有选定特征值的标签
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, edge_color='gray', arrowsize=20, arrowstyle='-|>', connectionstyle='arc3,rad=0.1', labels=labels)
    plt.title('6x4 Grid Graph with Selected Feature')
    plt.show()

if __name__ == '__main__':
    time_length = 6
    bin_length = 4
    direct = True
    adj_matrix = gen_adjmatrix(time_length, bin_length, direct)

    np.random.seed(0)
    features = np.random.rand(time_length* bin_length, 6)
    draw_grid_with_features(adj_matrix, time_length, bin_length, features, feature_index=0)
import numpy as np
import torch


def k_adjacency(A, k, with_self=False, self_factor=1):
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
        assert layout in ['nturgb+d', 'openpose', 'openpose_new', 'coco', 'coco_new', 'bones_seed_g1']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)
            ]
            self.center = 1
        elif layout == 'openpose_new':
            self.num_node = 20
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 18), (8, 18), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14), (18, 19), (19, 1)
            ]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        elif layout == 'coco_new':
            self.num_node = 20
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 17), (12, 17),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 19), (6, 19),
                (1, 0), (3, 1), (2, 0), (4, 2), (0, 19), (17, 18), (18, 19)
            ]
            self.center = 19
        elif layout == 'bones_seed_g1':
            # 35-DoF Unitree G1 robot skeleton
            # Nodes 0-5: root (translateX/Y/Z, rotateX/Y/Z)
            # Nodes 6-8: left_hip (pitch/roll/yaw)
            # Node 9: left_knee; Nodes 10-11: left_ankle (pitch/roll)
            # Nodes 12-14: right_hip; Node 15: right_knee; Nodes 16-17: right_ankle
            # Nodes 18-20: waist (yaw/roll/pitch)
            # Nodes 21-23: left_shoulder (pitch/roll/yaw)
            # Node 24: left_elbow; Nodes 25-27: left_wrist (roll/pitch/yaw)
            # Nodes 28-30: right_shoulder; Node 31: right_elbow; Nodes 32-34: right_wrist
            self.num_node = 35
            self.inward = [
                # root chain
                (1, 0), (2, 1), (3, 2), (4, 3), (5, 4),
                # root -> left hip -> knee -> ankle
                (6, 2), (7, 6), (8, 7), (9, 8), (10, 9), (11, 10),
                # root -> right hip -> knee -> ankle
                (12, 2), (13, 12), (14, 13), (15, 14), (16, 15), (17, 16),
                # root -> waist -> left shoulder -> elbow -> wrist
                (18, 2), (19, 18), (20, 19),
                (21, 20), (22, 21), (23, 22), (24, 23), (25, 24), (26, 25), (27, 26),
                # waist -> right shoulder -> elbow -> wrist
                (28, 20), (29, 28), (30, 29), (31, 30), (32, 31), (33, 32), (34, 33),
            ]
            self.center = 2  # root_translateZ
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off

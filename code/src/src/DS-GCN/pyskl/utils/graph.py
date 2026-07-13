import numpy as np
import torch


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
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
    # A is a 2D square array
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

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

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
        assert layout in ['openpose', 'nturgb+d', 'coco', 'bones_seed_bvh', 'bones_seed_g1']

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
            node_type = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,1,1,2,2]
            # combinations = np.array(np.meshgrid(node_type, node_type)).T.reshape(-1,2)
            self.node_type = node_type
            self.edge_type =np.zeros([self.num_node,self.num_node])
            index  = np.array(node_type).reshape(25,1)+1
            index = index*pow(-1,index)
            edge_type_index = np.dot(index, index.T)
            unique, _ =np.unique(edge_type_index,return_counts=True)
            for i in range(len(unique)):
                self.edge_type[edge_type_index==unique[i]]=i
            self.edge_type_num = unique
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
            node_type = [0,0,0,0,0,1,2,1,2,1,2,3,4,3,4,3,4]
            self.node_type = node_type
            self.edge_type =np.zeros([self.num_node,self.num_node])
            index  = np.array(node_type).reshape(self.num_node,1)+1
            index = index*pow(-1,index)
            edge_type_index = np.dot(index, index.T)
            unique, _ =np.unique(edge_type_index,return_counts=True)
            for i in range(len(unique)):
                self.edge_type[edge_type_index==unique[i]]=i
            self.edge_type_num = unique
        elif layout == 'bones_seed_bvh':
            # 24-joint BVH skeleton from BONES-SEED dataset.
            # Joints correspond to MAJOR_JOINT_CHANNELS in src/unveil.py
            # (sorted by channel index):
            # 0=Root_rot, 1=Hips_rot, 2=Spine1, 3=Spine2, 4=Chest,
            # 5=Neck1, 6=Neck2, 7=Head,
            # 8=LeftShoulder, 9=LeftArm, 10=LeftForeArm, 11=LeftHand,
            # 12=RightShoulder, 13=RightArm, 14=RightForeArm, 15=RightHand,
            # 16=LeftLeg, 17=LeftShin, 18=LeftFoot, 19=LeftToeBase,
            # 20=RightLeg, 21=RightShin, 22=RightFoot, 23=RightToeBase
            self.num_node = 24
            self.inward = [
                (1, 0),   # Hips <- Root
                (2, 1),   # Spine1 <- Hips
                (3, 2),   # Spine2 <- Spine1
                (4, 3),   # Chest <- Spine2
                (5, 4),   # Neck1 <- Chest
                (6, 5),   # Neck2 <- Neck1
                (7, 6),   # Head <- Neck2
                (8, 4),   # LeftShoulder <- Chest
                (9, 8),   # LeftArm <- LeftShoulder
                (10, 9),  # LeftForeArm <- LeftArm
                (11, 10), # LeftHand <- LeftForeArm
                (12, 4),  # RightShoulder <- Chest
                (13, 12), # RightArm <- RightShoulder
                (14, 13), # RightForeArm <- RightArm
                (15, 14), # RightHand <- RightForeArm
                (16, 0),  # LeftLeg <- Root
                (17, 16), # LeftShin <- LeftLeg
                (18, 17), # LeftFoot <- LeftShin
                (19, 18), # LeftToeBase <- LeftFoot
                (20, 0),  # RightLeg <- Root
                (21, 20), # RightShin <- RightLeg
                (22, 21), # RightFoot <- RightShin
                (23, 22), # RightToeBase <- RightFoot
            ]
            self.center = 0  # Root
            # Node types: 0=spine, 1=head/neck, 2=left_arm, 3=right_arm,
            #             4=left_leg, 5=right_leg
            node_type = [0, 0, 0, 0, 0, 1, 1, 1,
                         2, 2, 2, 2, 3, 3, 3, 3,
                         4, 4, 4, 4, 5, 5, 5, 5]
            self.node_type = node_type
            self.edge_type = np.zeros([self.num_node, self.num_node])
            index = np.array(node_type).reshape(self.num_node, 1) + 1
            index = index * pow(-1, index)
            edge_type_index = np.dot(index, index.T)
            unique, _ = np.unique(edge_type_index, return_counts=True)
            for i in range(len(unique)):
                self.edge_type[edge_type_index == unique[i]] = i
            self.edge_type_num = unique

        elif layout == 'bones_seed_g1':
            # 35-DoF G1 robot skeleton from BONES-SEED dataset.
            # Columns (after dropping Frame) in the G1 CSV:
            # 0-5:   root (translateX/Y/Z, rotateX/Y/Z)
            # 6-11:  left leg (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
            # 12-17: right leg (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
            # 18-20: waist (yaw, roll, pitch)
            # 21-27: left arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
            # 28-34: right arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
            self.num_node = 35
            inward = []
            # Intra-group chains (connect DoFs within same body segment)
            for i in range(0, 5):    inward.append((i + 1, i))    # root chain
            for i in range(6, 11):   inward.append((i + 1, i))    # left leg chain
            for i in range(12, 17):  inward.append((i + 1, i))    # right leg chain
            inward.append((19, 18)); inward.append((20, 19))       # waist chain
            for i in range(21, 27):  inward.append((i + 1, i))    # left arm chain
            for i in range(28, 34):  inward.append((i + 1, i))    # right arm chain
            # Inter-group (parent-child body part connections)
            inward.append((6, 5))    # left_hip_pitch  <- root_rotateZ (representative)
            inward.append((12, 5))   # right_hip_pitch <- root_rotateZ
            inward.append((18, 5))   # waist_yaw       <- root_rotateZ
            inward.append((21, 20))  # left_shoulder   <- waist_pitch
            inward.append((28, 20))  # right_shoulder  <- waist_pitch
            self.inward = inward
            self.center = 0  # root_translateX (root representative)
            # Node types: 0=root, 1=left_leg, 2=right_leg, 3=waist,
            #             4=left_arm, 5=right_arm
            node_type = (
                [0] * 6 +   # root (0-5)
                [1] * 6 +   # left leg (6-11)
                [2] * 6 +   # right leg (12-17)
                [3] * 3 +   # waist (18-20)
                [4] * 7 +   # left arm (21-27)
                [5] * 7     # right arm (28-34)
            )
            self.node_type = node_type
            self.edge_type = np.zeros([self.num_node, self.num_node])
            index = np.array(node_type).reshape(self.num_node, 1) + 1
            index = index * pow(-1, index)
            edge_type_index = np.dot(index, index.T)
            unique, _ = np.unique(edge_type_index, return_counts=True)
            for i in range(len(unique)):
                self.edge_type[edge_type_index == unique[i]] = i
            self.edge_type_num = unique

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

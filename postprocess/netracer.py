import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import morphology, generate_binary_structure
from skimage import morphology, transform
import heapq
from rtree import index
import copy
from tqdm import tqdm
import sys
import os
import time
import tifffile as tiff
import math
from pathlib import Path

sys.setrecursionlimit(1000000)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.dirname(CURRENT_DIR))

from swclib.data.swc_forest import SwcForest
from swclib.data.swc_node import SwcNode
from swclib.data.euclidean_point import Line, EuclideanPoint3D
from swclib.geometry.Obj3D import Point3D, Sphere, Cone
from swclib.image.swc2mask import setMarkWithSphere, setMarkWithCone

from utils.gpu import get_gpu_with_max_free_vram


def local_max(Im, wsize=3, thre=255 * 0.5):
    nZ, nY, nX = Im.shape
    suppress = np.zeros_like(Im)
    potential_points = np.where(Im > thre)
    num_points = len(potential_points[0])
    coordinates = []
    for i in tqdm(range(num_points), disable=True):
        z = potential_points[0][i]
        y = potential_points[1][i]
        x = potential_points[2][i]
        if wsize == 3:
            if x < 1 or y < 1 or z < 1 or x > nX - 2 or y > nY - 2 or z > nZ - 2:
                continue

            img_patch = Im[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2]
            if img_patch.max() == img_patch[1, 1, 1]:
                suppress[z, y, x] = 255
                coordinates.append([z, y, x, Im[z, y, x]])
        if wsize == 5:
            if x < 2 or y < 2 or z < 2 or x > nX - 3 or y > nY - 3 or z > nZ - 3:
                continue

            img_patch = Im[z - 2 : z + 3, y - 2 : y + 3, x - 2 : x + 3]
            if img_patch.max() == img_patch[2, 2, 2]:
                suppress[z, y, x] = 255
                coordinates.append([z, y, x, Im[z, y, x]])

        if wsize == 7:
            if x < 3 or y < 3 or z < 3 or x > nX - 4 or y > nY - 4 or z > nZ - 4:
                continue

            img_patch = Im[z - 3 : z + 4, y - 3 : y + 4, x - 3 : x + 4]
            if img_patch.max() == img_patch[3, 3, 3]:
                suppress[z, y, x] = 255
                coordinates.append([z, y, x, Im[z, y, x]])
    return suppress, coordinates


###############
#### Model ####
###############
class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, 1, padding=1, bias=True)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class up_conv_3d(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv_3d, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class down_conv_3d(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(down_conv_3d, self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class res_conv_block_3d(nn.Module):
    """
    Res Convolution Block
    """

    def __init__(self, out_ch):
        super(res_conv_block_3d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x) + x)
        return out


class PE_Net_3D(nn.Module):
    """
    Input: Original Image, Walked Image
    Output: Centerline, Point, Edge
    """

    def __init__(self, in_ch=1, out_ch=1, freeze_net=False):
        super(PE_Net_3D, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.conv_input = ConvReLU(in_ch, filters[0])

        self.Conv1 = res_conv_block_3d(filters[0])
        self.Down1 = down_conv_3d(filters[0], filters[1])

        self.Conv2 = res_conv_block_3d(filters[1])
        self.Down2 = down_conv_3d(filters[1], filters[2])

        self.Conv3 = res_conv_block_3d(filters[2])
        self.Down3 = down_conv_3d(filters[2], filters[3])

        self.Conv4 = res_conv_block_3d(filters[3])
        self.Down4 = down_conv_3d(filters[3], filters[4])

        self.Conv5_1 = res_conv_block_3d(filters[4])

        self.Up5 = up_conv_3d(filters[4], filters[3])
        self.Up_conv5 = res_conv_block_3d(filters[3])

        self.Up4 = up_conv_3d(filters[3], filters[2])
        self.Up_conv4 = res_conv_block_3d(filters[2])

        self.Up3 = up_conv_3d(filters[2], filters[1])
        self.Up_conv3 = res_conv_block_3d(filters[1])

        self.Up2 = up_conv_3d(filters[1], filters[0])
        self.Up_conv2 = res_conv_block_3d(filters[0])

        self.centerline_block = res_conv_block_3d(filters[0])

        self.Conv_centerline = nn.Sequential(
            res_conv_block_3d(filters[0]), res_conv_block_3d(filters[0])
        )
        self.Conv_centerline_out = nn.Conv3d(
            filters[0], out_ch, kernel_size=1, stride=1, padding=0
        )

        # walked path encoder
        self.walked_path_input = ConvReLU(in_ch, filters[0])

        # Point-Edge encoder
        self.point_edge_block = ConvReLU(filters[0] + filters[0], filters[0])

        self.Conv_edge = nn.Sequential(
            res_conv_block_3d(filters[0]), res_conv_block_3d(filters[0])
        )
        self.Conv_edge_out = nn.Sequential(
            res_conv_block_3d(filters[0]),
            nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0),
        )
        self.Conv_point = nn.Sequential(
            res_conv_block_3d(filters[0]),
            res_conv_block_3d(filters[0]),
            nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0),
        )

        self.active_Sigmoid = nn.Sigmoid()
        self.active_ReLU = nn.ReLU()

    def forward(self, input_image, walked_path=None):

        x = self.conv_input(input_image)
        e1 = self.Conv1(x)
        e1_d = self.Down1(e1)

        e2 = self.Conv2(e1_d)
        e2_d = self.Down2(e2)

        e3 = self.Conv3(e2_d)
        e3_d = self.Down3(e3)

        e4 = self.Conv4(e3_d)
        e4_d = self.Down4(e4)

        e5_2 = self.Conv5_1(e4_d)

        d5 = self.Up5(e5_2)
        d5 = torch.add(e4, d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.add(e3, d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.add(e2, d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.add(e1, d2)
        stage_fuse = self.Up_conv2(d2)

        centerline_fts = self.centerline_block(stage_fuse)

        # centerline encoder
        d0_centerline = self.Conv_centerline(centerline_fts)
        d0_centerline_out = self.Conv_centerline_out(d0_centerline)
        centerline_out = self.active_Sigmoid(d0_centerline_out)

        if walked_path is None:
            return centerline_out

        # walked path encoder
        walked_path_f = self.walked_path_input(walked_path)
        stage_fuse = torch.cat([centerline_fts, walked_path_f], dim=1)

        # Point-Edge encoder
        point_edge_fts = self.point_edge_block(stage_fuse)
        # Edge block
        edge_fts = self.Conv_edge(point_edge_fts)
        edge_fts_out = self.Conv_edge_out(edge_fts)
        edge_out = self.active_Sigmoid(edge_fts_out)

        # Point block
        point_fts_input = torch.add(walked_path_f, edge_fts)
        point_fts = self.Conv_point(point_fts_input)
        point_out = self.active_Sigmoid(point_fts)

        return centerline_out, point_out, edge_out


#######################
#### tracing tools ####
#######################
def gen_circle_gaussian_3d(
    size=[16, 64, 64],
    r=32.0,
    z_offset=0,
    y_offset=0,
    x_offset=0,
):
    z0 = size[0] // 2
    y0 = size[1] // 2
    x0 = size[2] // 2
    x0 += x_offset
    y0 += y_offset
    z0 += z_offset
    z, y, x = np.ogrid[: size[0], : size[1], : size[2]]
    image = 1 * np.exp(
        -(
            ((x - x0) ** 2 / (2 * r**2))
            + ((y - y0) ** 2 / (2 * r**2))
            + ((z - z0) ** 2 / (2 * r**2))
        )
    )
    return image.astype(np.float32)


def find_if_parent(node_A, node_B, k=3):
    # find if nodeA is nodeB's parent, k is num
    node_current = node_B
    node_target = node_A

    for i in range(k):
        # print(node_current, node_current.parent)
        if node_current.parent == node_target:
            return True
        else:
            # print(node_current, node_current.parent)
            node_current = node_current.parent

            # if node_current.get_id() == 1:
            # 	return True
            if node_current.is_virtual():  # 找到了根节点
                return False
    return False


def o_distance(point_A, point_B):
    distance = math.sqrt(
        (point_A[0] - point_B[0]) ** 2
        + (point_A[1] - point_B[1]) ** 2
        + (point_A[2] - point_B[2]) ** 2
    )
    return distance


def in_range(n, start, end=0):
    return start <= n <= end if end >= start else end <= n <= start


def cosine_similarity(vector_A, vector_B):
    numerator = 0
    denominator_A = 0
    denominator_B = 0
    for i in range(len(vector_A)):
        numerator = numerator + vector_A[i] * vector_B[i]
        denominator_A = denominator_A + vector_A[i] ** 2
        denominator_B = denominator_B + vector_B[i] ** 2
    result = numerator / np.sqrt(denominator_A + 1e-6) / np.sqrt(denominator_B + 1e-6)
    return result


def get_bounds(point_a, point_b, extra=0):
    """
    get bounding box of a segment
    Args:
        point_a: two points to identify the square
        point_b:
        extra: float, a threshold
    Return:
        res(tuple):
    """
    point_a = np.array(point_a.coord)
    point_b = np.array(point_b.coord)
    res = (np.where(point_a > point_b, point_b, point_a) - extra).tolist() + (
        np.where(point_a > point_b, point_a, point_b) + extra
    ).tolist()

    return tuple(res)


structure = generate_binary_structure(3, 1)


def find_path_edge(img, seed, end_point_list):
    img = img.astype(np.float32)
    img_max = np.max(img)
    img_min = np.min(img)
    range_ = img_max - img_min

    score_img = np.full_like(img, np.inf)
    score_img[seed[0]][seed[1]][seed[2]] = 1

    path_img = np.ones_like(score_img).astype(np.float32)
    path_img[seed[0]][seed[1]][seed[2]] = np.inf

    walked_img = np.full(
        (img.shape[0] + 2, img.shape[1] + 2, img.shape[2] + 2), np.inf, dtype=np.float32
    )
    walked_img[1:-1, 1:-1, 1:-1] = 1
    walked_img[seed[0] + 1][seed[1] + 1][seed[2] + 1] = np.inf

    seq_img = np.zeros_like(score_img).astype(np.float32)
    seq_len_img = np.zeros_like(score_img).astype(np.float32)

    k = 0
    next_node_pos = seed

    while k < 10000:
        z_temp = next_node_pos[0]
        y_temp = next_node_pos[1]
        x_temp = next_node_pos[2]

        structure_path_temp = (
            copy.deepcopy(
                walked_img[
                    z_temp - 1 + 1 : z_temp + 2 + 1,
                    y_temp - 1 + 1 : y_temp + 2 + 1,
                    x_temp - 1 + 1 : x_temp + 2 + 1,
                ]
            )
            * structure
        )
        next_node_list_temp = np.where(structure_path_temp == 1)

        for node_temp_id in range(len(next_node_list_temp[0])):
            z_ = next_node_list_temp[0][node_temp_id] + z_temp - 1
            y_ = next_node_list_temp[1][node_temp_id] + y_temp - 1
            x_ = next_node_list_temp[2][node_temp_id] + x_temp - 1
            score_current = score_img[z_][y_][x_]
            k_ = 10
            score_1 = np.exp(k_ * ((img[z_][y_][x_] - img_min) / range_)) - 1
            score_2 = (
                np.exp(k_ * ((img[z_temp][y_temp][x_temp] - img_min) / range_)) - 1
            )
            score_new = score_img[z_temp][y_temp][x_temp] + (score_1 + score_2) / 2
            score_img[z_][y_][x_] = min(score_current, score_new)
            walked_img[z_ + 1][y_ + 1][x_ + 1] = np.inf

        # 更新当前位置，已经遍历
        walked_img[z_temp + 1][y_temp + 1][x_temp + 1] = np.inf
        max_index = np.argmin(score_img * path_img)
        z_new, y_new, x_new = np.unravel_index(max_index, score_img.shape)
        path_img[z_new][y_new][x_new] = np.inf

        next_node_pos = [z_new, y_new, x_new]
        if next_node_pos in end_point_list:
            path_img[z_new][y_new][x_new] = np.inf
            break
        if score_img[z_new][y_new][x_new] == 255:
            break
        path_img[z_temp][y_temp][x_temp] = np.inf
        k = k + 1

    return path_img, score_img, seq_img, seq_len_img, next_node_pos


def dijkstra_path_edge(img, seed, end_point_list, k_exp=10, max_iter=2_000_000):
    img = img.astype(np.float32)
    img_min, img_max = float(img.min()), float(img.max())
    range_ = img_max - img_min
    if range_ == 0:
        range_ = 1e-12

    Z, Y, X = img.shape
    seed = tuple(seed)
    end_set = set(
        map(tuple, end_point_list)
    )  # list of list -> set of tuple，加速 & 稳定

    # 6邻域偏移：structure(3,1) 里除中心点外为 True 的位置
    neigh = np.array(np.where(structure)).T - 1  # shape (7,3), 包含(0,0,0)
    neigh = [
        tuple(d) for d in neigh if not (d[0] == 0 and d[1] == 0 and d[2] == 0)
    ]  # 去掉中心

    dist = np.full((Z, Y, X), np.inf, dtype=np.float32)
    visited = np.zeros((Z, Y, X), dtype=bool)

    dist[seed] = 0.0
    heap = [(0.0, seed[0], seed[1], seed[2])]

    def node_cost(z, y, x):
        # 你原来的 exp 代价：exp(k*norm)-1
        norm = (img[z, y, x] - img_min) / range_
        return np.exp(k_exp * norm) - 1.0

    it = 0
    hit = None

    while heap and it < max_iter:
        d, z, y, x = heapq.heappop(heap)
        if visited[z, y, x]:
            continue
        visited[z, y, x] = True
        if (z, y, x) in end_set:
            hit = (z, y, x)
            break

        c2 = node_cost(z, y, x)
        for dz, dy, dx in neigh:
            zz, yy, xx = z + dz, y + dy, x + dx
            if zz < 0 or zz >= Z or xx < 0 or xx >= X or yy < 0 or yy >= Y:
                continue
            if visited[zz, yy, xx]:
                continue
            c1 = node_cost(zz, yy, xx)
            nd = d + 0.5 * (c1 + c2)
            if nd < dist[zz, yy, xx]:
                dist[zz, yy, xx] = nd
                heapq.heappush(heap, (float(nd), zz, yy, xx))
        it += 1

    return hit


def tracing_strategy(current_node_dict, tree_new_rtree, tree_new_idedge_dict, verbose):
    node_id = current_node_dict["node_id"]
    z_temp_new_ave = current_node_dict["node_z"]
    x_temp_new_ave = current_node_dict["node_x"]
    y_temp_new_ave = current_node_dict["node_y"]
    # node_r = 2.0 * 1.5
    node_r = 1.0 * 1.5

    parent_node = current_node_dict["parent_node"]
    node_exist = current_node_dict["node_exist"]

    if node_id == 1:  # 第一步忽略
        node_range = node_r
        distance_to_nearby_branch = np.inf

    elif (
        node_id == 2
    ):  # 第二步（由于无法edge_match_utils.get_nearby_edges计算距离，因此单独设置）
        node_new = EuclideanPoint3D(
            center=[x_temp_new_ave, y_temp_new_ave, z_temp_new_ave]
        )
        node_parent = EuclideanPoint3D(
            center=[parent_node[0], parent_node[1], parent_node[2]]
        )
        distance_to_nearby_branch = node_new.distance_to_point(node_parent)
        node_range = node_r
    else:
        son_node_temp = SwcNode(
            nid=1,
            ntype=0,
            coord=EuclideanPoint3D(
                center=[x_temp_new_ave, y_temp_new_ave, z_temp_new_ave]
            ),
            radius=node_r,
        )
        node_temp_list = get_nearby_edges(
            rtree=tree_new_rtree,
            point=son_node_temp,
            id_edge_dict=tree_new_idedge_dict,
            threshold=node_r * 3,
        )  # 三倍半径内的枝干
        if len(node_temp_list) == 0:
            node_range = node_r
            # distance_to_nearby_branch = -1 # 可能产生了大跳跃，因此停止
            distance_to_nearby_branch = np.inf  # 可能产生了大跳跃
        else:
            distance_to_nearby_branch = node_temp_list[0][1]
            node_range = node_r
            if (
                son_node_temp.coord[0] == parent_node.coord[0]
                and son_node_temp.coord[1] == parent_node.coord[1]
                and son_node_temp.coord[2] == parent_node.coord[2]
            ):
                node_range = np.inf

    # 中止判断条件
    boundary_th = 128  # should be adjust
    if node_exist < boundary_th:
        if verbose:
            print("存在越界", node_exist)
        end_tracing = True
        end_tracing_code = "01"
        return end_tracing, end_tracing_code
    elif distance_to_nearby_branch < node_range:
        if distance_to_nearby_branch == -1:
            if verbose:
                print("this may be a large gap")
            end_tracing_code = "02"
            # end_tracing = True
        else:
            if verbose:
                print("this position is traced")
            end_tracing_code = "03"
            # end_tracing = False
        end_tracing = True
        return end_tracing, end_tracing_code
    else:
        end_tracing_code = "00"
        end_tracing = False
        return end_tracing, end_tracing_code


# 根据网络架构调整
def get_pos_image_3d(image, node_list, pos, SHAPE):
    z_half = SHAPE[0] // 2
    y_half = SHAPE[1] // 2
    x_half = SHAPE[2] // 2
    pos_z, pos_y, pos_x = pos

    node_img = image[
        pos_z - z_half : pos_z + z_half,
        pos_y - y_half : pos_y + y_half,
        pos_x - x_half : pos_x + x_half,
    ].copy()

    mark = np.zeros((SHAPE[0], SHAPE[1], SHAPE[2]))
    mark_shape = (SHAPE[0], SHAPE[1], SHAPE[2])

    start_point = [z_half, y_half, x_half]  # (x, y) 起点坐标
    if len(node_list) < 2:
        setMarkWithSphere(
            mark, Sphere(Point3D(*start_point), 1), mark_shape
        )
    else:
        for i in range(2, len(node_list) + 1):
            end_point = copy.deepcopy(node_list[-i])

            end_point[0] = end_point[0] - node_list[-1][0] + SHAPE[0] // 2
            end_point[1] = end_point[1] - node_list[-1][1] + SHAPE[1] // 2
            end_point[2] = end_point[2] - node_list[-1][2] + SHAPE[2] // 2

            setMarkWithCone(
                mark,
                Cone(Point3D(*start_point), 1, Point3D(*end_point), 1),
                mark_shape,
            )
            start_point = end_point

    img_walk = np.array(mark).astype(np.uint8)

    return node_img, img_walk


def get_network_predict_vecroad_3d(
    org_skl_temp,
    image,
    node_list_walked,
    image_walk,
    SHAPE,
    model,
    device
):
    predicted_time = 0

    image = np.sqrt(copy.deepcopy(image)) / 255  # * 2 - 1
    image_walk = copy.deepcopy(image_walk) / 255

    data_transform = torchvision.transforms.Compose([])

    # seq_len, _,_,_ = image.shape
    image_tensor = np.zeros(dtype=np.float32, shape=[1, 2, *SHAPE])
    image_tensor[0, 0, :, :, :] = copy.deepcopy(image[0])
    image_tensor[0, 1, :, :, :] = copy.deepcopy(image_walk[0])

    image_tensor_input = data_transform(image_tensor)
    test_loader = torch.utils.data.DataLoader(image_tensor_input, batch_size=1)

    model.eval()
    torch.no_grad()  # to increase the validation process uses less memory

    for x_batch in test_loader:
        batch_input = x_batch.to(device)

        img_org, img_walk = batch_input[0][0], batch_input[0][1]

        img_org = img_org.reshape(1, 1, SHAPE[0], SHAPE[1], SHAPE[2])
        img_walk = img_walk.reshape(1, 1, SHAPE[0], SHAPE[1], SHAPE[2])

        pre_time_a = time.time()
        # 网络预测
        _, y_point_pred, y_edge_pred = model(img_org, img_walk)  # v7-m2
        pre_time_b = time.time()
        predicted_time += pre_time_b - pre_time_a

        # Point预处理
        point_temp_mask = (
            gen_circle_gaussian_3d(
                size=SHAPE, r=2.0, z_offset=0, x_offset=0, y_offset=0
            )
            * 255
        )
        point_temp_mask[point_temp_mask <= 128] = 0
        point_temp_mask[point_temp_mask > 128] = 255
        point_temp_mask = point_temp_mask / 255

        pred_tb = y_point_pred.cpu().detach().numpy()
        pred_tb = pred_tb.reshape(SHAPE[0], SHAPE[1], SHAPE[2])
        mask = np.zeros_like(pred_tb)
        mask[1:-1, 4:-4, 4:-4] = 1
        pred_tb = mask * pred_tb * 255

        pred_tb = (1 - point_temp_mask) * pred_tb

        # Edge预处理
        pred_edge = y_edge_pred.cpu().detach().numpy() * 255
        pred_edge = pred_edge.reshape(
            SHAPE[0], SHAPE[1], SHAPE[2]
        ).astype(np.uint8)

        # 选取结点保存到next_node_pos
        next_node_pos_patch = []
        next_node_pos = []
        if np.sum(pred_tb) == 0:
            max_depth = 0
            max_row = 0
            max_col = 0
            next_node_pos.append(
                [
                    max_depth + node_list_walked[-1][0],
                    max_row + node_list_walked[-1][1],
                    max_col + node_list_walked[-1][2],
                ]
            )
        else:
            point_th = np.max(pred_tb) * 0.5
            pred_tb_temp = copy.deepcopy(pred_tb)
            while np.max(pred_tb_temp) > point_th:
                max_index = np.argmax(pred_tb_temp)
                max_depth, max_row, max_col = np.unravel_index(
                    max_index, pred_tb_temp.shape
                )
                next_node_pos_patch.append([max_depth, max_row, max_col])
                max_depth = max_depth - SHAPE[0] // 2
                max_row = max_row - SHAPE[1] // 2
                max_col = max_col - SHAPE[2] // 2
                point_temp_mask = (
                    gen_circle_gaussian_3d(
                        size=[16, 64, 64],
                        r=2.0,
                        z_offset=max_depth,
                        y_offset=max_row,
                        x_offset=max_col,
                    )
                    * 255
                )
                point_temp_mask[point_temp_mask <= 50] = 0
                point_temp_mask[point_temp_mask > 50] = 255
                point_temp_mask = point_temp_mask / 255
                point_temp_mask = point_temp_mask.astype(np.uint8)
                pred_tb_temp = (1 - point_temp_mask) * pred_tb_temp

        center_z = SHAPE[0] // 2
        center_y = SHAPE[1] // 2
        center_x = SHAPE[2] // 2
        seed = [center_z, center_y, center_x]

        # path_img, score_img , _ ,_, next_node_pos_ = find_path_edge(pred_edge, seed, next_node_pos_patch)
        next_node_pos_ = dijkstra_path_edge(
            pred_edge, seed, next_node_pos_patch
        )  # 根据边预测找next_node_pos_patch中最优的点

        max_depth = next_node_pos_[0] - SHAPE[0] // 2
        max_row = next_node_pos_[1] - SHAPE[1] // 2
        max_col = next_node_pos_[2] - SHAPE[2] // 2

        next_node_pos_ = [0, 0, 0]
        next_node_pos_[0] = max_depth + node_list_walked[-1][0]
        next_node_pos_[1] = max_row + node_list_walked[-1][1]
        next_node_pos_[2] = max_col + node_list_walked[-1][2]

        next_node_pos.append(next_node_pos_)

        next_node_pos_ = [0, 0, 0]
        next_node_pos_[0] = -max_depth + node_list_walked[-1][0]
        next_node_pos_[1] = -max_row + node_list_walked[-1][1]
        next_node_pos_[2] = -max_col + node_list_walked[-1][2]

        next_node_pos.append(next_node_pos_)

        next_node_pos_final = []
        next_node_exist_final = []

        for node_pos_temp in next_node_pos:
            z_temp_new = node_pos_temp[0]
            y_temp_new = node_pos_temp[1]
            x_temp_new = node_pos_temp[2]

            node_pos_next = [round(z_temp_new), round(y_temp_new), round(x_temp_new)]

            z_temp_new_ave = z_temp_new
            y_temp_new_ave = y_temp_new
            x_temp_new_ave = x_temp_new

            node_pos_next = [
                round(z_temp_new_ave),
                round(y_temp_new_ave),
                round(x_temp_new_ave),
            ]
            # node_exist_next = np.max(org_skl_temp[node_pos_next[0]-2:node_pos_next[0]+2, node_pos_next[1]-2:node_pos_next[1]+2,node_pos_next[2]-2:node_pos_next[2]+2])
            node_exist_next = np.max(
                org_skl_temp[
                    node_pos_next[0] - 5 : node_pos_next[0] + 5,
                    node_pos_next[1] - 5 : node_pos_next[1] + 5,
                    node_pos_next[2] - 5 : node_pos_next[2] + 5,
                ]
            )
            next_node_pos_final.append(node_pos_next)
            next_node_exist_final.append(node_exist_next)
        return next_node_exist_final, next_node_pos_final


def tracing_strategy_single_vecroad_3d_test(
    org_image,
    org_skl_temp,
    seed_node,
    seed_node_dict,
    tree_new,
    model,
    device,
    SHAPE,
    r_tree_info,
    verbose=False,
):
    tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]
    end_tracing = False
    current_node_dict = copy.deepcopy(seed_node_dict)
    node_pool = []
    node_pool.append(current_node_dict)

    steps = 1
    begin_node_id = tree_new.size() + 1
    # 开始追踪
    while len(node_pool) != 0:
        if verbose:
            print(begin_node_id, "--------------------------------", steps)

        # 从node_pool中取一个节点
        current_node_dict = node_pool.pop()
        end_tracing, end_tracing_code = tracing_strategy(
            current_node_dict, tree_new_rtree, tree_new_idedge_dict, verbose
        )
        if end_tracing == True:
            if verbose:
                print(f"当前点不满足条件，跳过，换下一个点:{end_tracing_code}")
            continue
        else:
            current_z_temp = round(current_node_dict["node_z"])
            current_x_temp = round(current_node_dict["node_x"])
            current_y_temp = round(current_node_dict["node_y"])

            node_r = current_node_dict["node_r"]
            node_list_walked = current_node_dict["node_list_walked"]
            current_vector = current_node_dict["direction_vector"]
            parent_node = current_node_dict["parent_node"]

            # 将当前结点加入追踪结果中
            if steps == 1:
                son_node = seed_node
                tree_new.id_set.add(seed_node_dict["node_id"])
                tree_new.get_node_list(update=True)
            else:
                son_node = SwcNode(
                    nid=begin_node_id,
                    ntype=0,
                    coord=[round(current_x_temp, 3), round(current_y_temp, 3), round(current_z_temp, 3)],
                    radius=round(node_r, 3),
                )
                tree_new.add_child(parent_node, son_node)
                tree_new.get_node_list(update=True)
                tree_new_rtree.insert(
                    son_node.nid,
                    get_bounds(son_node, son_node.parent, extra=node_r * 1.5),
                )
                tree_new_idedge_dict[son_node.nid] = tuple(
                    [son_node, son_node.parent]
                )

            steps = steps + 1
            begin_node_id = begin_node_id + 1

            # 未来点的预测
            node_pos = [current_z_temp, current_y_temp, current_x_temp]
            seed_node_img, seed_node_img_walked = get_pos_image_3d(
                org_image, node_list_walked, node_pos, SHAPE
            )
            seed_node_img = seed_node_img.reshape(1, *SHAPE)
            seed_node_img_walked = seed_node_img_walked.reshape(1, *SHAPE)
            exist, next_node_pos = get_network_predict_vecroad_3d(
                org_skl_temp,
                seed_node_img,
                node_list_walked,
                seed_node_img_walked,
                SHAPE,
                model,
                device,
            )

            # 按照方向相似度,对结点重排序
            cos_sim_list = []
            for node_pos_temp in next_node_pos:
                z_temp_new = node_pos_temp[0]
                y_temp_new = node_pos_temp[1]
                x_temp_new = node_pos_temp[2]
                next_vector = [
                    z_temp_new - current_z_temp,
                    y_temp_new - current_y_temp,
                    x_temp_new - current_x_temp,
                ]
                cos_sim = cosine_similarity(current_vector, next_vector)
                cos_sim_list.append(cos_sim)

            def sort_and_return_index(lst):
                sorted_lst = sorted(lst)
                index_lst = [i[0] for i in sorted(enumerate(lst), key=lambda x: x[1])]
                return sorted_lst, index_lst

            _, cos_sim_sorted_list_index = sort_and_return_index(
                cos_sim_list
            )

            # 按照相似度低到高排序，依次添加========
            for node_temp_id in cos_sim_sorted_list_index:
                # 如果偏离完全相反，则跳过
                if cos_sim_list[node_temp_id] < 0:
                    # print('方向相反，已跳过')
                    continue

                z_temp_new_ave = next_node_pos[node_temp_id][0]
                y_temp_new_ave = next_node_pos[node_temp_id][1]
                x_temp_new_ave = next_node_pos[node_temp_id][2]
                node_exist = exist[node_temp_id]

                node_pos_next = [
                    round(z_temp_new_ave),
                    round(y_temp_new_ave),
                    round(x_temp_new_ave),
                ]
                if verbose:
                    print(
                        "预测点位置：",
                        node_temp_id + 1,
                        "|",
                        round(z_temp_new_ave) - SHAPE[0],
                        round(y_temp_new_ave) - SHAPE[1],
                        round(x_temp_new_ave) - SHAPE[2],
                    )

                # 判断是否在图像区域外
                if (
                    in_range(
                        round(z_temp_new_ave),
                        SHAPE[0] // 2,
                        org_image.shape[0] - SHAPE[0] // 2,
                    )
                    is False
                    or in_range(
                        round(y_temp_new_ave),
                        SHAPE[1] // 2,
                        org_image.shape[1] - SHAPE[1] // 2,
                    )
                    is False
                    or in_range(
                        round(x_temp_new_ave),
                        SHAPE[2] // 2,
                        org_image.shape[2] - SHAPE[2] // 2,
                    )
                    is False
                ):
                    if verbose:
                        print("exceed the bound")
                    continue
                # 判断是否离父节点太远，要求10倍半径内
                dis_temp = o_distance(node_pos, node_pos_next)
                if dis_temp > 10 * node_r:
                    if verbose:
                        print("too far from its parent!")
                    continue
                # 判断是否和父节点位置一样
                if (
                    round(z_temp_new_ave) == current_z_temp
                    and round(y_temp_new_ave) == current_y_temp
                    and round(x_temp_new_ave) == current_x_temp
                ):
                    if verbose:
                        print("same as parent node!")
                    continue

                # 更新结点信息
                # 更新node_list_walked信息, 历史信息长度为10
                node_list_walked_len = 15
                node_list_walked_temp = copy.deepcopy(node_list_walked)
                if len(node_list_walked_temp) < node_list_walked_len:
                    node_list_walked_temp.append(
                        [z_temp_new_ave, y_temp_new_ave, x_temp_new_ave]
                    )
                else:
                    node_list_walked_temp.remove(node_list_walked_temp[0])
                    node_list_walked_temp.append(
                        [z_temp_new_ave, y_temp_new_ave, x_temp_new_ave]
                    )

                temp_node_dict = {"node_id": begin_node_id}
                temp_node_dict["node_id"] = begin_node_id
                temp_node_dict["node_z"] = z_temp_new_ave
                temp_node_dict["node_x"] = x_temp_new_ave
                temp_node_dict["node_y"] = y_temp_new_ave

                temp_node_dict["node_r"] = 2.0
                temp_node_dict["direction_vector"] = [
                    z_temp_new_ave - current_z_temp,
                    y_temp_new_ave - current_y_temp,
                    x_temp_new_ave - current_x_temp,
                ]

                temp_node_dict["node_p_id"] = current_node_dict["node_id"]

                temp_node_dict["parent_node"] = son_node
                temp_node_dict["node_exist"] = node_exist
                temp_node_dict["node_list_walked"] = node_list_walked_temp

                node_pool.append(temp_node_dict)

    r_tree_info = [tree_new_rtree, tree_new_idedge_dict]
    return tree_new, r_tree_info

def get_nearby_edges(rtree, point, id_edge_dict, threshold, not_self=False, debug=False):
    '''
    find the close enough edges of a node base on rtree
    sorted by distance
    Args:
        rtree(): an rtree describe the target edge set
        point: the point to get nearby edges
        id_edge_dict:map between id and line tuple(edge)
        threshold: the ceiling of the distance between point and edge
        not_self: exclude self, used in overlap detect
        debug:
    Returns:
         a list of tuple(edge, dis). Sorted according to distance to the point
    level: 1
    '''
    point_box = (point[0] - threshold, point[1] - threshold, point[2] - threshold,
                 point[0] + threshold, point[1] + threshold, point[2] + threshold)
    hits = list(rtree.intersection(point_box))
    nearby_edges = []

    for h in hits:
        line_tuple = id_edge_dict[h]
        if debug:
            print("\npoint = id{} poi{}, line_a = id{} poi{}, line_b = id{} poi{}".format(
                point.nid, point.coord,
                line_tuple[0].nid, line_tuple[0].coord,
                line_tuple[1].nid, line_tuple[1].coord)
            )
        e_point = point.coord

        # if two sides of line_tuple is in the same position, ignore this line
        if line_tuple[0].coord.distance(line_tuple[1].coord) == 0:
            continue

        new_d = e_point.distance(Line(e_node_1=line_tuple[0].coord, e_node_2=line_tuple[1].coord))
        if not_self and new_d == 0:
            continue
        nearby_edges.append(tuple([line_tuple, new_d]))
    nearby_edges.sort(key=lambda x: x[1])
    return nearby_edges

"""
tracing
"""
def trace_netracer(raw, seg, out_path, verbose=False):
    if isinstance(raw, str) or isinstance(raw, Path):
        stack_img = tiff.imread(raw).astype(np.float32)
    else:
        stack_img = raw.astype(np.float32)
    if isinstance(seg, str) or isinstance(seg, Path):
        stack_skl = tiff.imread(seg).astype(np.float32)
    else:
        stack_skl = seg.astype(np.float32)
    
    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    checkpoint_path = os.path.join(CURRENT_DIR, "weights", "netracer-weight-C2.pth")
    SHAPE = [16, 64, 64]

    test_patch_height = SHAPE[1]
    test_patch_width = SHAPE[2]
    test_patch_depth = SHAPE[0]

    stride_height = 48
    stride_width = 48
    stride_depth = 8

    #######################################################
    #               build the model
    #######################################################
    best_i, best_free, info_list = get_gpu_with_max_free_vram()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{best_i}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PE_Net_3D().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()

    #######################################################
    #                tracing the image                    #
    #######################################################
    # find seed
    if stack_skl.max() == 1:
        stack_skl = stack_skl * 255
    assert (stack_skl.max()==0) or (stack_skl.max() > 1 and stack_skl.max() <= 255), "the value of skeleton image should be in the range of (0, 255]"
    _, candidate_sup = local_max(stack_skl, wsize=3, thre=0.5 * 255)
    candidate_file = np.array(candidate_sup)
    candidate_file = candidate_file[np.argsort(-candidate_file[:, -1])]
    seed_list = candidate_file[:, :3].T
    seed_list[0] += SHAPE[0]
    seed_list[1] += SHAPE[1]
    seed_list[2] += SHAPE[2]

    # ===============resize the image=====================
    d, h, w = stack_img.shape
    d_new = ((d - (test_patch_depth - stride_depth)) // (stride_depth) + 1) * (
        stride_depth
    ) + (test_patch_depth - stride_depth)
    h_new = ((h - (test_patch_height - stride_height)) // (stride_height) + 1) * (
        stride_height
    ) + (test_patch_height - stride_height)
    w_new = ((w - (test_patch_width - stride_width)) // (stride_width) + 1) * (
        stride_width
    ) + (test_patch_width - stride_width)
    stack_img_new = np.zeros([d_new, h_new, w_new], dtype=np.float32)
    stack_skl_new = np.zeros([d_new, h_new, w_new], dtype=np.uint8)

    stack_img_new[0:d, 0:h, 0:w] = copy.deepcopy(stack_img)
    stack_skl_new[0:d, 0:h, 0:w] = copy.deepcopy(stack_skl)

    org_img_shape = stack_img_new.shape
    org_img_temp = np.zeros(
        [
            org_img_shape[0] + 2 * SHAPE[0],
            org_img_shape[1] + 2 * SHAPE[1],
            org_img_shape[2] + 2 * SHAPE[2],
        ]
    )
    org_skl_temp = np.zeros(
        [
            org_img_shape[0] + 2 * SHAPE[0],
            org_img_shape[1] + 2 * SHAPE[1],
            org_img_shape[2] + 2 * SHAPE[2],
        ]
    )

    org_img_temp[
        1 * SHAPE[0] : 1 * SHAPE[0] + org_img_shape[0],
        1 * SHAPE[1] : 1 * SHAPE[1] + org_img_shape[1],
        1 * SHAPE[2] : 1 * SHAPE[2] + org_img_shape[2],
    ] = copy.deepcopy(stack_img_new)
    org_skl_temp[
        1 * SHAPE[0] : 1 * SHAPE[0] + org_img_shape[0],
        1 * SHAPE[1] : 1 * SHAPE[1] + org_img_shape[1],
        1 * SHAPE[2] : 1 * SHAPE[2] + org_img_shape[2],
    ] = copy.deepcopy(stack_skl_new)

    seed_list_flag = np.ones(len(seed_list[0]))
    if verbose:
        print("seed node number: %d" % (seed_list_flag.shape[0]))

    # build tree
    tree_new = SwcForest()
    virtual_root = SwcNode(nid=-1, coord=EuclideanPoint3D([0, 0, 0]))
    tree_new.add_tree(virtual_root)
    tree_new_idedge_dict = {}
    p = index.Property()
    p.dimension = 3
    tree_new_rtree = index.Index(properties=p)
    r_tree_info = [tree_new_rtree, tree_new_idedge_dict]

    for i in range(seed_list_flag.shape[0]):
        if i == 0:
            glob_node_id = 1
        else:
            glob_node_id = max(tree_new.id_set) + 1

        if verbose:
            print(
                "------------------- tracing ----------------------",
                out_path.split("/")[-1],
                i + 1,
                " / ",
                seed_list_flag.shape[0],
            )
        seed_node_z = seed_list[0][i]
        seed_node_y = seed_list[1][i]
        seed_node_x = seed_list[2][i]

        # init seed node
        seed_node_dict = {
            "node_id": glob_node_id,
            "node_z": seed_node_z,
            "node_x": seed_node_x,
            "node_y": seed_node_y,
            "node_r": 2,
            "node_p_id": -1,
            "parent_node": tree_new.roots[0],
            "node_exist": 255,
            "node_list_walked": [[seed_node_z, seed_node_y, seed_node_x]],
        }
        seed_node_dict["direction_vector"] = [0, 0, 0]

        seed_node = SwcNode(
            nid=seed_node_dict["node_id"],
            ntype=0,
            coord=[seed_node_dict["node_x"],seed_node_dict["node_y"],seed_node_dict["node_z"]],
            radius=round(seed_node_dict["node_r"], 3),
        )
        node_temp_list = get_nearby_edges(
            rtree=tree_new_rtree,
            point=seed_node,
            id_edge_dict=tree_new_idedge_dict,
            threshold=2,
        )
        if len(node_temp_list) != 0:
            if verbose:
                print("this seed is already traced")
            continue

        # process the seed node
        seed_node.parent = seed_node_dict["parent_node"]

        # start tracing
        tree_new, r_tree_info = tracing_strategy_single_vecroad_3d_test(
            org_img_temp,
            org_skl_temp,
            seed_node,
            seed_node_dict,
            tree_new,
            model,
            device,
            SHAPE,
            r_tree_info,
            verbose
        )
        tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]

    tree_new.relocation([-SHAPE[2], -SHAPE[1], -SHAPE[0]])
    tree_new.remove_node(virtual_root)
    tree_new.save_to_file(out_path)


if __name__ == "__main__":
    img_path = "/data1/yangzekang/neuron/neuron-trace/examples/real/id1_f3_b0_raw.tif"
    seg_path = "/data1/yangzekang/neuron/neuron-trace/examples/real/id1_f3_b0_mask.tif"
    out_path = (
        "/data1/yangzekang/neuron/neuron-trace/examples/real/id1_f3_b0_pred-netracer-v2.swc"
    )

    trace_netracer(img_path, seg_path, out_path, verbose=True)

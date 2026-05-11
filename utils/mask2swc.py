import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
import tifffile as tiff


def skeleton_to_tree(skl, dim=3):
    assert len(skl.shape) == dim
    nodes = np.argwhere(skl == 1)
    voxel_idx = {tuple(v): i for i, v in enumerate(nodes)}
    G = nx.Graph()
    for i, n in enumerate(nodes):
        G.add_node(i, pos=n)
    neighbor_shifts = np.array(
        [
            [i, j, k]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for k in [-1, 0, 1]
            if not (i == 0 and j == 0 and k == 0)
        ]
    )
    for i, n in enumerate(nodes):
        for shift in neighbor_shifts:
            neighbor = tuple(n + shift)
            if neighbor in voxel_idx:
                j = voxel_idx[neighbor]
                if not G.has_edge(i, j):
                    weight = np.linalg.norm(n - nodes[j])
                    G.add_edge(i, j, weight=weight)
    T = nx.minimum_spanning_tree(G, weight="weight")
    return T


def mask_to_tree(img, dim=3):
    assert len(img.shape) == dim
    skeleton = skeletonize(img)
    return skeleton_to_tree(skeleton)


def mask_to_swc(img, dim=3, scale=(0.35, 0.35, 1), save_path="tmp.swc"):
    T = mask_to_tree(img, dim)
    components = list(nx.connected_components(T))
    swc_lines = []
    node_id_map = {}
    swc_counter = 1
    for comp_nodes in components:
        tree = T.subgraph(comp_nodes)
        degrees = tree.degree()
        root_node = max(degrees, key=lambda x: x[1])[0]

        parent = {root_node: -1}
        visited = set([root_node])
        queue = [root_node]

        while queue:
            current = queue.pop(0)
            for neighbor in tree.neighbors(current):
                if neighbor not in visited:
                    parent[neighbor] = current
                    visited.add(neighbor)
                    queue.append(neighbor)
        for node in parent:
            coord = tree.nodes[node]["pos"]
            pid = parent[node]
            pid_swc = node_id_map.get(pid, -1)

            swc_lines.append(
                f"{swc_counter} 0 {coord[2]*scale[0]} {coord[1]*scale[1]} {coord[0]*scale[2]} 1.0 {pid_swc}\n"
            )
            node_id_map[node] = swc_counter
            swc_counter += 1
    with open(save_path, "w") as f:
        f.writelines(swc_lines)

import numpy as np
import networkx as nx


def swc_file_to_graph(swc_file):
    """
    Convert an SWC file to a NetworkX graph.

    SWC format: n T x y z R P
        n: node id
        T: type
        x,y,z: position
        R: radius
        P: parent node id (-1 for root)
    """
    G = nx.Graph()
    with open(swc_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            n, T, x, y, z, R, P = parts
            n, T, P = int(n), int(T), int(P)
            x, y, z, R = map(float, (x, y, z, R))
            G.add_node(n, pos=np.array([x, y]))
            if P != -1:
                G.add_edge(n, P)
    return G


def swc_to_graph(swcs):
    G = nx.Graph()
    for swc in swcs:
        n, T, x, y, z, R, P = swc
        n, T, P = int(n), int(T), int(P)
        x, y, z, R = map(float, (x, y, z, R))
        G.add_node(n, pos=np.array([z, y, x]))
        if P != -1:
            G.add_edge(n, P)
    return G

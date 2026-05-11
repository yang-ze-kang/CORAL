import numpy as np
from sklearn.neighbors import KDTree
import networkx as nx


def is_empty(G):
    return len(G.edges()) == 0


def node_degree(G, node):
    return len(G.edges(node))


def is_intersection(G, node):
    return node_degree(G, node) > 2


def is_end_point(G, node):
    return node_degree(G, node) == 1


def is_control_nodes(G, node):
    return is_intersection(G, node) or is_end_point(G, node)


def insert_nodes_in_edge(G, s, t, nodes, nodes_pos):

    G_ = G  # .copy()

    # reorder nodes positions
    def distance(idx):
        return (G_.nodes[s]["pos"][0] - nodes_pos[idx][0]) ** 2 + (
            G_.nodes[s]["pos"][1] - nodes_pos[idx][1]
        ) ** 2

    idxs = list(range(len(nodes)))
    idxs.sort(key=lambda idx: distance(idx))

    G_.remove_edge(s, t)
    G_.add_node(nodes[idxs[0]], pos=nodes_pos[idxs[0]], snapped=True)
    G_.add_edge(s, nodes[idxs[0]])
    for i_1, i in zip(idxs[:-1], idxs[1:]):
        G_.add_node(nodes[i], pos=nodes_pos[i], snapped=True)
        G_.add_edge(nodes[i_1], nodes[i])
    G_.add_edge(nodes[idxs[-1]], t)

    return G_


def closest_points_on_segments(X, P, Q):
    """
    Computes the closest point on a segment to a point
    for all points and all segments

    Parameters
    ----------
    X : numpy.ndarray (N,M)
        points in the space
    P and Q : numpy.ndarray (O,M)
        points defining the start and end of the segments

    Return
    ------
    S : numpy.ndarray (N,O,M)
        the closest points to X on the segments
    D : numpy.ndarray (N,O)
        distance from the points X to the closests on the segments
    id : numpy.int (N,O)
        0 if S=P, 1 if S=Q and None if on the segment
    """
    assert len(X) != 0
    assert len(P) != 0
    assert len(Q) != 0

    N, M = X.shape

    Q_P = (Q - P)[None]
    X_P = X[:, None, :] - P[None, :, :]
    lambdas = np.sum(X_P * Q_P, axis=2) / (np.sum(Q_P * Q_P, axis=2) + 1e-12)  # [N,O]

    id = np.array([[None] * len(P)] * len(X))  # [N,O]
    id[lambdas <= 0] = 0
    id[lambdas >= 1] = 1

    lambdas = np.repeat(lambdas[:, :, None], M, axis=2)  # [N,O,M]
    S = P[None] + lambdas * Q_P  # [N,O,M]
    np.putmask(S, lambdas <= 0, np.repeat(P[None], N, axis=0))
    np.putmask(S, lambdas >= 1, np.repeat(Q[None], N, axis=0))

    D = np.linalg.norm(S - X[:, None], axis=2)

    return S, D, id


def snap_points_to_graph(G, points, th_existing=10, th_snap=25, inplace=False):

    name_new_node = lambda i: str(i) + "_snapped"

    if inplace:
        G_ = G
    else:
        G_ = G.copy()

    dim = points.shape[-1]
    points_ = np.reshape(points, (-1, dim))
    edges = list(G.edges())
    s_nodes = np.array([G.nodes[s]["pos"] for s, t in edges])
    t_nodes = np.array([G.nodes[t]["pos"] for s, t in edges])

    S, D, id = closest_points_on_segments(points_, s_nodes, t_nodes)

    # find the edges where to snap the new points
    to_snap = {}
    correspondences = []
    for idx_point, point in enumerate(points_):

        idx_closest_edge = D[idx_point].argmin()
        dist = D[idx_point, idx_closest_edge]

        s, t = edges[idx_closest_edge]

        # do not snap the point if it is too far form any edge
        if dist < th_snap:

            # do not create an additional node if the closest point is the
            # starting or ending nodes of the edge
            if id[idx_point, idx_closest_edge] == 0:
                correspondences.append(s)
            elif id[idx_point, idx_closest_edge] == 1:
                correspondences.append(t)
            else:
                # If one between the starting or ending nodes is very close to the point
                # do not create an additional node in the graph.
                if np.linalg.norm(s_nodes[idx_closest_edge] - point) < th_existing:
                    correspondences.append(s)
                elif np.linalg.norm(t_nodes[idx_closest_edge] - point) < th_existing:
                    correspondences.append(t)
                else:
                    if idx_closest_edge not in to_snap:
                        to_snap[idx_closest_edge] = []
                    to_snap[idx_closest_edge].append(idx_point)
                    correspondences.append(name_new_node(idx_point))

        else:
            correspondences.append(None)

    # modify the edges
    for idx_closest_edge, idxs_points in to_snap.items():

        s, t = edges[idx_closest_edge]

        new_nodes = [name_new_node(i) for i in idxs_points]
        new_nodes_pos = [S[idx_point, idx_closest_edge] for idx_point in idxs_points]
        G_ = insert_nodes_in_edge(G_, s, t, new_nodes, new_nodes_pos)

    return G_, correspondences


def matching_with_snapping(
    junctions_g, junctions_pos_g, G, H, th_existing=1, th_snap=25, alpha=100
):

    # snap the node of graph G into graph H if they are sufficiently close
    H_snap, corresps = snap_points_to_graph(
        H, junctions_pos_g, th_existing=th_existing, th_snap=th_snap
    )

    # get control nodes
    junctions_h = [
        n for n in H_snap.nodes() if is_control_nodes(H_snap, n) or n in corresps
    ]
    junctions_pos_h = np.array([H_snap.nodes[n]["pos"] for n in junctions_h])

    # find some candidates control points to match
    dists_all, idxs_all = KDTree(junctions_pos_h).query(junctions_pos_g, k=5)
    candidates_all = []
    for dists, idxs in zip(dists_all, idxs_all):
        candidates_all.append([(d, junctions_h[i]) for d, i in zip(dists, idxs)])

    # find a node in H that correspond at best to each node in G
    matches = {}
    matches_pos = []
    seen = []
    for node_g, candidates in zip(junctions_g, candidates_all):

        order_g = len(G.edges(node_g))

        best_cost_h = np.inf
        best_node_h = None
        # try to find the best control points
        for dist, node_h in candidates:

            # consider close candidates only
            if dist <= th_snap:

                # during the matching, we add some priviledges
                # to the nodes with similar order.
                order_h = len(H_snap.edges(node_h))

                cost = dist + alpha * np.abs(order_g - order_h)
                if cost < best_cost_h and node_h not in seen:
                    best_cost_h = cost
                    best_node_h = node_h

        matches[node_g] = best_node_h
        seen.append(best_node_h)

    return matches, H_snap


def twoway_matching(G, H, th_existing=1, th_snap=25, alpha=100):

    junctions_g = [n for n in G.nodes() if is_control_nodes(G, n)]
    junctions_g_pos = np.array([G.nodes[n]["pos"] for n in junctions_g])
    matches_g, H_snap = matching_with_snapping(
        junctions_g, junctions_g_pos, G, H, th_existing, th_snap, alpha
    )

    junctions_h = [n for n in H.nodes() if is_control_nodes(H, n)]
    junctions_hg = [n for n in junctions_h if n not in matches_g.values()]
    if len(junctions_hg) > 0:
        junctions_hg_pos = np.array([H.nodes[n]["pos"] for n in junctions_hg])

        matches_hg, G_snap = matching_with_snapping(
            junctions_hg, junctions_hg_pos, H, G, th_existing, th_snap, alpha
        )
    else:
        matches_hg = {}
        G_snap = G

    return (
        matches_g,
        matches_hg,
        G_snap,
        H_snap,
    )  # matches from graph G to graph H with snapped nodes as well


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-12)


def compute_scores(tp, ap, pp):

    precision = tp / (pp + 1e-12)
    recall = tp / (ap + 1e-12)
    f1 = f1_score(precision, recall)

    return f1, precision, recall


def opt_j(G_gt, G_pred, th_existing=1, th_snap=25, alpha=100):
    """
    OPT-J metric

    Leonardo Citraro, Mateusz Kozinski, Pascal Fua
    Towards Reliable Evaluation of Road Network Reconstructions
    ECCV 2020

    Parameters
    ----------
    G_gt : networkx object
        ground-truth graph
    G_pred : networkx object
        reconstructed graph
    th_snap : float
        a point is snapped into the graph if its distance from the
        closest edge is less than th_snap
    th_existing : float
        during the snapping prcedure, an additional node is inserted into an edge only if
        none of endpoints of the edge are within th_existing
    alpha : float
        parameter that encourage matching two nodes that have similar order

    Return
    ------
    matches_g : dict
        matched nodes from G_gt to G_pred_snap
    matches_hg : dict
        remaining matches from G_pred to G_gt_snap
    g_gt_snap : networkx object
        G_gt with the added nodes
    g_pred_snap : networkx object
        G_pred with the added nodes
    """

    if is_empty(G_gt):
        raise ValueError("Ground-truth graph is empty!")

    if is_empty(G_pred):
        print("!! Predicted graph is empty !!")
        f1, precision, recall = 0, 0, 0
        tp, pp, ap = 0, 0, 0
        matches_g, matches_hg = {}, {}
        g_gt_snap, g_pred_snap = G_gt, G_pred
        return (
            f1,
            precision,
            recall,
            tp,
            pp,
            ap,
            matches_g,
            matches_hg,
            g_gt_snap,
            g_pred_snap,
        )

    matches_g, matches_hg, g_gt_snap, g_pred_snap = twoway_matching(
        G_gt, G_pred, th_existing, th_snap, alpha=alpha
    )
    tp = 0
    pp = 0
    ap = 0
    for node_gt, node_pred in matches_g.items():

        order_gt = len(g_gt_snap.edges(node_gt))
        if node_pred is not None:
            order_pred = len(g_pred_snap.edges(node_pred))
        else:
            order_pred = 0

        tp += np.minimum(order_gt, order_pred)
        pp += order_pred
        ap += order_gt

    for node_pred, node_gt in matches_hg.items():
        if node_gt is not None:
            order_gt = len(g_gt_snap.edges(node_gt))
        else:
            order_gt = 0

        order_pred = len(g_pred_snap.edges(node_pred))

        tp += np.minimum(order_gt, order_pred)
        pp += order_pred
        ap += order_gt

    f1, precision, recall = compute_scores(tp, ap, pp)

    return (
        f1,
        precision,
        recall,
        tp,
        pp,
        ap,
        matches_g,
        matches_hg,
        g_gt_snap,
        g_pred_snap,
    )

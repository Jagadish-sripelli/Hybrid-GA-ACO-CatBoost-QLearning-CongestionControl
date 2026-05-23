"""
network_graph.py
----------------
Network topology generation utilities for grid, random, and scale-free
topologies as used in the paper (Section 7.2).
"""

import numpy as np
import networkx as nx
import random


def create_topology(topology_type: str, num_nodes: int, seed: int = 42) -> nx.DiGraph:
    """
    Create a directed weighted network graph.

    Parameters
    ----------
    topology_type : str
        One of 'grid', 'random', 'scale_free'
    num_nodes : int
        Number of nodes in the network
    seed : int
        Random seed for reproducibility

    Returns
    -------
    nx.DiGraph
        Directed graph with edge attributes:
        bandwidth, delay, loss_prob, queue_occupancy, energy
    """
    random.seed(seed)
    np.random.seed(seed)

    if topology_type == 'grid':
        G = _create_grid(num_nodes, seed)
    elif topology_type == 'random':
        G = _create_random(num_nodes, seed)
    elif topology_type == 'scale_free':
        G = _create_scale_free(num_nodes, seed)
    else:
        raise ValueError(f"Unknown topology: {topology_type}. "
                         f"Choose from grid, random, scale_free")

    G = _assign_edge_attributes(G, seed)
    return G


def _create_grid(num_nodes: int, seed: int) -> nx.DiGraph:
    """Square grid topology."""
    side = int(np.ceil(np.sqrt(num_nodes)))
    G_undirected = nx.grid_2d_graph(side, side)
    # Relabel nodes to integers
    mapping = {node: i for i, node in enumerate(G_undirected.nodes())}
    G_undirected = nx.relabel_nodes(G_undirected, mapping)
    # Keep only num_nodes nodes
    nodes_to_keep = list(range(num_nodes))
    G_undirected = G_undirected.subgraph(nodes_to_keep).copy()
    return nx.DiGraph(G_undirected)


def _create_random(num_nodes: int, seed: int) -> nx.DiGraph:
    """Erdos-Renyi random topology."""
    p = 0.15  # edge probability
    G_undirected = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
    # Ensure connectivity
    while not nx.is_connected(G_undirected):
        G_undirected = nx.erdos_renyi_graph(num_nodes, p, seed=seed + 1)
    return nx.DiGraph(G_undirected)


def _create_scale_free(num_nodes: int, seed: int) -> nx.DiGraph:
    """Barabasi-Albert scale-free topology."""
    m = 3  # edges to attach from new node
    G_undirected = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    return nx.DiGraph(G_undirected)


def _assign_edge_attributes(G: nx.DiGraph, seed: int) -> nx.DiGraph:
    """
    Assign realistic QoS attributes to each edge.

    Attributes (per paper Section 7.2):
        bandwidth  : Mbps  [1.0 – 100.0]
        delay      : ms    [1.0 – 50.0]
        loss_prob  : [0.0 – 0.15]
        queue_occ  : queue occupancy ratio [0.0 – 1.0]
        energy     : Joules per packet [0.01 – 0.5]
    """
    rng = np.random.RandomState(seed)
    for u, v in G.edges():
        G[u][v]['bandwidth']  = rng.uniform(1.0, 100.0)
        G[u][v]['delay']      = rng.uniform(1.0, 50.0)
        G[u][v]['loss_prob']  = rng.uniform(0.0, 0.15)
        G[u][v]['queue_occ']  = rng.uniform(0.0, 0.8)
        G[u][v]['energy']     = rng.uniform(0.01, 0.5)
        G[u][v]['flow']       = 0.0
    return G


def update_edge_state(G: nx.DiGraph,
                      congested_edges: list,
                      rng: np.random.RandomState) -> nx.DiGraph:
    """
    Dynamically update edge states to simulate traffic dynamics.

    Parameters
    ----------
    G : nx.DiGraph
    congested_edges : list of (u, v) tuples
    rng : numpy RandomState

    Returns
    -------
    nx.DiGraph with updated queue_occ and loss_prob
    """
    for u, v in G.edges():
        if (u, v) in congested_edges:
            G[u][v]['queue_occ']  = min(1.0, G[u][v]['queue_occ'] + rng.uniform(0.1, 0.3))
            G[u][v]['loss_prob']  = min(1.0, G[u][v]['loss_prob'] + rng.uniform(0.05, 0.15))
        else:
            G[u][v]['queue_occ']  = max(0.0, G[u][v]['queue_occ'] - rng.uniform(0.0, 0.1))
            G[u][v]['loss_prob']  = max(0.0, G[u][v]['loss_prob'] - rng.uniform(0.0, 0.05))
    return G


def get_all_paths(G: nx.DiGraph, source: int, target: int,
                  cutoff: int = 5) -> list:
    """Return up to cutoff simple paths from source to target."""
    try:
        paths = list(nx.all_simple_paths(G, source, target, cutoff=cutoff))
        return paths[:10]  # cap at 10 candidates
    except nx.NetworkXNoPath:
        return []


def compute_path_qos(G: nx.DiGraph, path: list) -> dict:
    """
    Compute aggregate QoS metrics for a given path.

    Returns
    -------
    dict with keys: bandwidth, delay, loss_prob, queue_occ, energy
    """
    if len(path) < 2:
        return {}

    bandwidth   = float('inf')
    delay       = 0.0
    loss_prob   = 0.0
    queue_occ   = 0.0
    energy      = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        bandwidth = min(bandwidth, G[u][v]['bandwidth'])
        delay    += G[u][v]['delay']
        loss_prob = 1 - (1 - loss_prob) * (1 - G[u][v]['loss_prob'])
        queue_occ = max(queue_occ, G[u][v]['queue_occ'])
        energy   += G[u][v]['energy']

    return {
        'bandwidth':  bandwidth,
        'delay':      delay,
        'loss_prob':  loss_prob,
        'queue_occ':  queue_occ,
        'energy':     energy,
    }

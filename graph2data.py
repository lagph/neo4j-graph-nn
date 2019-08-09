from typing import Dict, List, Tuple, Callable, Union
from torch_geometric.data import Data
from py2neo import Node, Relationship
from py2neo.database import Cursor
import torch


def build_pytorch_geometric_data(
        matches: Cursor,
        target_key: str,
        node_featurizer: Callable[[Node], torch.Tensor],
        edge_featrizer: Callable[[Relationship], torch.Tensor]) -> Data:
    node_idx_map: Dict[Node, int] = {}
    edge_idx: Tuple[List[int], List[int]] = ([], [])
    edges: List[Relationship] = []
    current_node_idx = 0
    for match in matches:
        num_nodes = 0
        num_edges = 0
        for value in match.values():
            if isinstance(value, Node):
                if value not in node_idx_map.keys():
                    node_idx_map[value] = current_node_idx
                    current_node_idx += 1
                edge_idx[num_nodes].append(node_idx_map[value])
                num_nodes += 1
            elif isinstance(value, Relationship):
                num_edges += 1
                edges.append(value)
            else:
                raise ValueError("Failed to build pytorch_geometric data.\n"
                                 "matches must contain only Nodes and Relationships, "
                                 "but just saw a {}.".format(type(value)))
        if num_nodes != 2:
            raise ValueError("Failed to build pytorch_geometric data.\n"
                             "matches must contain two Nodes per match, "
                             "but just saw a match with {} node(s).".format(num_nodes))
        if num_edges < 1:
            raise ValueError("Failed to build pytorch_geometric data.\n"
                             "matches must contain at least one Relationship per match, "
                             "but just saw a match that didn't conatin any.")
        # if there's more than one edge between the nodes, create the extra edges
        for _ in range(num_edges - 1):
            for i in (0, 1):
                edge_idx[i].append(edge_idx[i][-1])
    edge_attr: List[Union[None, torch.Tensor]] = [None for _ in range(len(edges))]
    for i, edge in enumerate(edges):
        edge_attr[i] = edge_featrizer(edge)
    x: List[Union[None, torch.Tensor]] = [None for _ in range(len(node_idx_map))]
    y: List[float] = [0 for _ in range(len(node_idx_map))]
    for node, i in node_idx_map.items():
        x[i] = node_featurizer(node)
        try:
            y[i] = node[target_key]
        except KeyError as e:
            print("Hit a KeyError while getting the target in build_pytorch_geometric_data.\n"
                  "Either the target_key, {}, is incorrect or the graph contains corrupted nodes.\n"
                  "Node that caused the error: {}".format(target_key, node))
            raise e
    edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_idx_tensor, edge_attr=edge_attr)

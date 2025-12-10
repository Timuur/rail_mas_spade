from typing import List, Tuple

import networkx as nx


def build_graph(edges: List[dict]) -> nx.Graph:
    g = nx.Graph()
    for e in edges:
        u, v = e["u"], e["v"]
        g.add_edge(
            u,
            v,
            distance_km=float(e["distance_km"]),
            speed_kmh=float(e.get("speed_kmh", 60)),
        )
    return g


def path_stats(g: nx.Graph, path: List[str]) -> Tuple[float, float]:
    d = t = 0.0
    for i in range(len(path) - 1):
        data = g.get_edge_data(path[i], path[i + 1])
        dist = float(data["distance_km"])
        speed = float(data.get("speed_kmh", 60))
        d += dist
        t += dist / speed
    return round(d, 2), round(t, 2)


def shortest_path(g: nx.Graph, a: str, b: str) -> List[str]:
    return nx.shortest_path(g, a, b, weight=lambda u, v, d: d["distance_km"])
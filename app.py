# app.py

import random
import time
import heapq
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="MST Algorithm Comparison",
    layout="wide"
)

st.title("Minimum Spanning Tree Algorithm Comparison")
st.caption("Same random graph calculated using Kruskal, Prim, and Borůvka")


# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Graph Settings")

num_nodes = st.sidebar.slider("Number of Nodes", 6, 40, 12)
edge_probability = st.sidebar.slider("Edge Probability", 0.2, 1.0, 0.45)
max_weight = st.sidebar.slider("Maximum Edge Weight", 5, 100, 30)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

show_labels = st.sidebar.checkbox("Show Edge Weights", value=True)


# --------------------------------------------------
# Generate One Connected Random Graph
# --------------------------------------------------
def generate_connected_random_graph(n, p, max_w, seed_value):
    """
    Generates one connected undirected weighted graph.
    The same graph is used for all MST algorithms.
    """
    current_seed = seed_value

    while True:
        G = nx.gnp_random_graph(n=n, p=p, seed=current_seed, directed=False)

        if nx.is_connected(G):
            random.seed(current_seed)

            for u, v in G.edges():
                G[u][v]["weight"] = random.randint(1, max_w)

            return G

        current_seed += 1


G = generate_connected_random_graph(
    n=num_nodes,
    p=edge_probability,
    max_w=max_weight,
    seed_value=seed
)


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def graph_total_weight(G):
    return sum(data["weight"] for _, _, data in G.edges(data=True))


def mst_total_weight(mst_edges):
    return sum(weight for _, _, weight in mst_edges)


def create_mst_graph(original_graph, mst_edges):
    mst = nx.Graph()
    mst.add_nodes_from(original_graph.nodes())

    for u, v, w in mst_edges:
        mst.add_edge(u, v, weight=w)

    return mst


# --------------------------------------------------
# Union-Find for Kruskal and Boruvka
# --------------------------------------------------
class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        return True


# --------------------------------------------------
# Kruskal Algorithm
# --------------------------------------------------
def kruskal_mst(G):
    start_time = time.perf_counter()

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append((data["weight"], u, v))

    edges.sort()

    uf = UnionFind(G.nodes())
    mst_edges = []
    checked_edges = 0

    for weight, u, v in edges:
        checked_edges += 1

        if uf.union(u, v):
            mst_edges.append((u, v, weight))

        if len(mst_edges) == G.number_of_nodes() - 1:
            break

    end_time = time.perf_counter()

    return {
        "algorithm": "Kruskal",
        "mst_edges": mst_edges,
        "mst_weight": mst_total_weight(mst_edges),
        "checked_edges": checked_edges,
        "runtime": end_time - start_time,
        "complexity": "O(E log E)",
        "strategy": "Sort all edges, then add the smallest edge that does not create a cycle"
    }


# --------------------------------------------------
# Prim Algorithm
# --------------------------------------------------
def prim_mst(G, start_node=0):
    start_time = time.perf_counter()

    visited = set()
    min_heap = []
    mst_edges = []
    checked_edges = 0

    visited.add(start_node)

    for neighbor in G.neighbors(start_node):
        weight = G[start_node][neighbor]["weight"]
        heapq.heappush(min_heap, (weight, start_node, neighbor))

    while min_heap and len(mst_edges) < G.number_of_nodes() - 1:
        weight, u, v = heapq.heappop(min_heap)
        checked_edges += 1

        if v in visited:
            continue

        visited.add(v)
        mst_edges.append((u, v, weight))

        for neighbor in G.neighbors(v):
            if neighbor not in visited:
                next_weight = G[v][neighbor]["weight"]
                heapq.heappush(min_heap, (next_weight, v, neighbor))

    end_time = time.perf_counter()

    return {
        "algorithm": "Prim",
        "mst_edges": mst_edges,
        "mst_weight": mst_total_weight(mst_edges),
        "checked_edges": checked_edges,
        "runtime": end_time - start_time,
        "complexity": "O(E log V)",
        "strategy": "Start from one node and keep adding the cheapest edge to a new node"
    }


# --------------------------------------------------
# Boruvka Algorithm
# --------------------------------------------------
def boruvka_mst(G):
    start_time = time.perf_counter()

    uf = UnionFind(G.nodes())
    mst_edges = []
    checked_edges = 0
    components = G.number_of_nodes()

    while components > 1:
        cheapest = {}

        for u, v, data in G.edges(data=True):
            checked_edges += 1
            weight = data["weight"]

            set_u = uf.find(u)
            set_v = uf.find(v)

            if set_u == set_v:
                continue

            if set_u not in cheapest or weight < cheapest[set_u][2]:
                cheapest[set_u] = (u, v, weight)

            if set_v not in cheapest or weight < cheapest[set_v][2]:
                cheapest[set_v] = (u, v, weight)

        merged_this_round = 0

        for edge in cheapest.values():
            u, v, weight = edge

            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                components -= 1
                merged_this_round += 1

        if merged_this_round == 0:
            break

    end_time = time.perf_counter()

    return {
        "algorithm": "Boruvka",
        "mst_edges": mst_edges,
        "mst_weight": mst_total_weight(mst_edges),
        "checked_edges": checked_edges,
        "runtime": end_time - start_time,
        "complexity": "O(E log V)",
        "strategy": "Each component selects its cheapest outgoing edge, then components merge"
    }


# --------------------------------------------------
# Run All Algorithms on the SAME Graph
# --------------------------------------------------
kruskal_result = kruskal_mst(G)
prim_result = prim_mst(G, start_node=0)
boruvka_result = boruvka_mst(G)

results = [kruskal_result, prim_result, boruvka_result]


# --------------------------------------------------
# Draw Graph
# --------------------------------------------------
def draw_graph(original_graph, mst_edges=None, title="Graph"):
    pos = nx.spring_layout(original_graph, seed=seed)

    fig, ax = plt.subplots(figsize=(9, 6))

    nx.draw_networkx_nodes(
        original_graph,
        pos,
        ax=ax,
        node_size=650
    )

    nx.draw_networkx_labels(
        original_graph,
        pos,
        ax=ax,
        font_size=10
    )

    nx.draw_networkx_edges(
        original_graph,
        pos,
        ax=ax,
        width=1,
        alpha=0.25
    )

    if mst_edges is not None:
        mst_graph = create_mst_graph(original_graph, mst_edges)

        nx.draw_networkx_edges(
            mst_graph,
            pos,
            ax=ax,
            width=3
        )

    if show_labels:
        edge_labels = nx.get_edge_attributes(original_graph, "weight")
        nx.draw_networkx_edge_labels(
            original_graph,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=8
        )

    ax.set_title(title)
    ax.axis("off")

    return fig


# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------
original_weight = graph_total_weight(G)
mst_weight = kruskal_result["mst_weight"]
saving = original_weight - mst_weight

st.header("1. One Random Weighted Graph")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Nodes", G.number_of_nodes())
col2.metric("Edges", G.number_of_edges())
col3.metric("Original Graph Weight", original_weight)
col4.metric("MST Weight", mst_weight)

st.success(
    f"The same random graph is used for all algorithms. "
    f"The MST reduces total connection cost by {saving}."
)


# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Original Random Weighted Graph")
    st.pyplot(draw_graph(G, title="Original Graph"))

with right_col:
    selected_algorithm = st.selectbox(
        "Choose MST Result to Display",
        ["Kruskal", "Prim", "Boruvka"]
    )

    selected_result = next(
        result for result in results if result["algorithm"] == selected_algorithm
    )

    st.subheader(f"MST Produced by {selected_algorithm}")
    st.pyplot(
        draw_graph(
            G,
            mst_edges=selected_result["mst_edges"],
            title=f"{selected_algorithm} MST"
        )
    )


# --------------------------------------------------
# Comparison Table
# --------------------------------------------------
st.header("2. Same Graph Calculated Using Different Algorithms")

comparison_df = pd.DataFrame([
    {
        "Algorithm": result["algorithm"],
        "Strategy": result["strategy"],
        "MST Weight": result["mst_weight"],
        "Edges in MST": len(result["mst_edges"]),
        "Checked Edges": result["checked_edges"],
        "Runtime (seconds)": round(result["runtime"], 8),
        "Time Complexity": result["complexity"]
    }
    for result in results
])

st.dataframe(comparison_df, width="stretch")

st.markdown("""
### Main Point

Kruskal, Prim, and Boruvka are different algorithms, but on the same graph they all produce the same MST weight.
This shows that the MST is the same objective, but each algorithm reaches it in a different way.
""")


# --------------------------------------------------
# Runtime Bar Chart
# --------------------------------------------------
st.header("3. Runtime Comparison")

runtime_df = pd.DataFrame({
    "Algorithm": [result["algorithm"] for result in results],
    "Runtime": [result["runtime"] for result in results]
})

st.bar_chart(runtime_df.set_index("Algorithm"))


# --------------------------------------------------
# Checked Edges Bar Chart
# --------------------------------------------------
st.header("4. Edge Checking Comparison")

checked_df = pd.DataFrame({
    "Algorithm": [result["algorithm"] for result in results],
    "Checked Edges": [result["checked_edges"] for result in results]
})

st.bar_chart(checked_df.set_index("Algorithm"))


# --------------------------------------------------
# MST Edge Table
# --------------------------------------------------
st.header("5. MST Edges Selected")

edge_table = pd.DataFrame(
    selected_result["mst_edges"],
    columns=["Node U", "Node V", "Weight"]
)

st.write(f"Edges selected by {selected_algorithm}:")
st.dataframe(edge_table, width="stretch")


# --------------------------------------------------
# Research Paper Connection
# --------------------------------------------------
st.header("6. Connection to the Research Paper")

st.markdown("""
The classical MST algorithms above are deterministic.

They do not use random bits.

The research paper is about randomized MST algorithms.
The main question is not whether we can find a different MST.

The main question is:

> Can we find the same correct MST efficiently while using far fewer random bits?

Older randomized MST algorithms, such as the Karger-Klein-Tarjan algorithm, use random sampling to reduce the graph and achieve fast expected runtime.

The Pettie-Ramachandran paper shows that the amount of randomness can be reduced dramatically while preserving efficient expected performance.
""")


randomness_df = pd.DataFrame([
    {
        "Approach": "Classical MST",
        "Example": "Kruskal / Prim / Boruvka",
        "Random Bits": "0",
        "Purpose": "Deterministic greedy MST construction"
    },
    {
        "Approach": "Older Randomized MST",
        "Example": "Karger-Klein-Tarjan",
        "Random Bits": "High",
        "Purpose": "Use sampling to reduce graph size"
    },
    {
        "Approach": "Reduced-Randomness MST",
        "Example": "Pettie-Ramachandran Paper",
        "Random Bits": "Much lower",
        "Purpose": "Keep efficiency while reducing randomness"
    }
])

st.dataframe(randomness_df, width="stretch")


# --------------------------------------------------
# Presentation Script
# --------------------------------------------------
st.header("7. What I Will Say During Presentation")

st.markdown("""
1. This dashboard first creates one random weighted graph.
2. Then the exact same graph is solved using Kruskal, Prim, and Boruvka.
3. All three algorithms should produce the same MST weight.
4. The difference is not the final MST, but how each algorithm reaches it.
5. This helps us understand why MST is a core graph problem.
6. The paper goes one level deeper: it studies randomized MST algorithms.
7. The paper's contribution is reducing the amount of randomness needed while keeping efficient expected runtime.
""")

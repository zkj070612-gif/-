"""D 题(网络科学/运筹)常用图算法一页通。
依赖: networkx numpy matplotlib"""
import networkx as nx
import matplotlib.pyplot as plt
from plot_style import use_paper_style, save, PALETTE

# 建图: 边带权重(距离/时间/费用) 和 容量
edges = [("A","B",4,10), ("A","C",2,8), ("B","C",5,4), ("B","D",10,6),
         ("C","E",3,9), ("E","D",4,10), ("D","F",11,12), ("E","F",8,5)]
G = nx.Graph()
for u, v, w, cap in edges:
    G.add_edge(u, v, weight=w, capacity=cap)

# 1. 最短路
path = nx.shortest_path(G, "A", "F", weight="weight")
print("shortest path A->F:", path, "length =", nx.shortest_path_length(G, "A", "F", weight="weight"))

# 2. 最小生成树 (用最少成本把所有点连起来)
mst = nx.minimum_spanning_tree(G, weight="weight")
print("MST edges:", sorted(mst.edges()), "total =", mst.size(weight="weight"))

# 3. 最大流 / 最小割 (网络瓶颈在哪)
flow_value, _ = nx.maximum_flow(G.to_directed(), "A", "F", capacity="capacity")
cut_value, (S, T) = nx.minimum_cut(G.to_directed(), "A", "F", capacity="capacity")
print("max flow A->F:", flow_value, "| min cut separates", sorted(S), "from", sorted(T))

# 4. 节点重要性 (哪个节点是关键枢纽)
bc = nx.betweenness_centrality(G, weight="weight")
pr = nx.pagerank(G, weight="capacity")
for n in sorted(G, key=bc.get, reverse=True):
    print(f"  {n}: betweenness={bc[n]:.3f}  pagerank={pr[n]:.3f}  degree={G.degree(n)}")

# 5. 鲁棒性: 删掉最关键的节点后网络效率掉多少 (D 题"韧性"类问题的标准做法)
eff0 = nx.global_efficiency(G)
hub = max(bc, key=bc.get)
H = G.copy(); H.remove_node(hub)
print(f"remove hub {hub}: efficiency {eff0:.3f} -> {nx.global_efficiency(H):.3f}, connected = {nx.is_connected(H)}")

# 6. 画图: 节点大小表示重要性, 高亮最短路
use_paper_style()
fig, ax = plt.subplots(figsize=(6, 4))
pos = nx.spring_layout(G, seed=3)
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#BBBBBB", width=1.5)
nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(zip(path, path[1:])), edge_color=PALETTE[1], width=3)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=PALETTE[0], node_size=[400 + 2500 * bc[n] for n in G])
nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=8)
ax.set_title("Network: node size = betweenness, orange = shortest path A-F")
ax.axis("off")
save(fig, "fig_network")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Configuration
FILE_PATH = "/path/to/data/themerl_network_all_post_types.csv"
TOP_N_REPLIED = 10

# Colour & Shape Maps
reply_type_colors = {
    'Reply to self': '#87ceeb',
    'Reply to other': '#799F79',
    'Reply in own thread': '#877B9E',
}

post_type_shapes = {
    'Reply': 'o',
    'Original': 's',
    'Quote': '^',

}

# Load & Build
def load_and_build_graph(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')

    # Thread size mapping
    thread_sizes = df['root_post_id'].value_counts().to_dict()
    df['thread_size'] = df['root_post_id'].map(thread_sizes)

    # URI to post_id mapping
    uri_to_post_id = dict(zip(df['uri'], df['post_id']))

    # Graph creation
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(
            row['uri'],
            post_id=row['post_id'],
            reply_type=row.get('reply_type', 'Unknown'),
            post_type=row.get('post_type', 'Reply'),
            parent_post_id=row.get('parent_post_id'),
            root_post_id=row.get('root_post_id'),
            thread_size=row.get('thread_size', 1)
        )

    for _, row in df[df['post_type'] == 'Reply'].iterrows():
        parent = row['parent_post_id']
        child = row['uri']
        if pd.notna(parent):
            if parent not in G.nodes:
                # Add missing parent node
                G.add_node(
                    parent,
                    post_id='(external)',
                    reply_type='Unknown',
                    post_type='External',
                    parent_post_id=None,
                    root_post_id=None,
                    thread_size=0
                )
            G.add_edge(parent, child, reply_type=row.get('reply_type', 'Unknown'))

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Full reply type breakdown
    print("\n== Reply Type Breakdown ==")
    reply_type_counts = df['reply_type'].fillna('Unknown').value_counts()
    for reply_type, count in reply_type_counts.items():
        print(f"{reply_type}: {count}")

    # Top active threads by root
    print("\n== Top 10 Most Active Threads ==")
    top_threads = df['root_post_id'].value_counts().head(10)
    for uri, count in top_threads.items():
        post_id = uri_to_post_id.get(uri, 'Unknown')
        print(f"{post_id}: {count} replies")

    return G, df, uri_to_post_id

# Longest Thread
def find_longest_thread(G):
    longest = {'root': None, 'depth': 0, 'path': []}

    def dfs(node, path):
        children = list(G.successors(node))
        if not children:
            return path
        max_path = path
        for child in children:
            new_path = dfs(child, path + [child])
            if len(new_path) > len(max_path):
                max_path = new_path
        return max_path

    for node in G.nodes:
        if G.in_degree(node) == 0:
            path = dfs(node, [node])
            if len(path) > longest['depth']:
                longest = {'root': node, 'depth': len(path), 'path': path}

    print("\n== Longest Thread ==")
    print(f"Root Post: {G.nodes[longest['root']]['post_id']}")
    print(f"Depth: {longest['depth']}")
    print("Path:", [G.nodes[n]['post_id'] for n in longest['path']])
    return longest['path']

# Subgraph Drawing
def draw_subgraph(G, nodes, title):
    subG = G.subgraph(nodes).copy()
    pos = nx.spring_layout(subG, seed=42, k=0.3)
    plt.figure(figsize=(14, 10))
    node_colors = [reply_type_colors.get(subG.nodes[n]['reply_type'], '#B4B3B3') for n in subG.nodes]
    node_sizes = [300 for _ in subG.nodes]
    labels = {n: subG.nodes[n]['post_id'] for n in subG.nodes}
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    nx.draw_networkx_edges(subG, pos, edge_color='#979595', arrows=True, arrowsize=10, width=1)
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Full Thread Visualisation
def draw_cross_engagement(G, df):
    pos = nx.spring_layout(G, seed=42, k=0.2, iterations=200)
    plt.figure(figsize=(28, 22))

    # Group nodes by shape
    shape_groups = {}
    for node, data in G.nodes(data=True):
        shape = post_type_shapes.get(data['post_type'], 'o')
        shape_groups.setdefault(shape, []).append(node)

    for shape, nodes in shape_groups.items():
        node_colors = [reply_type_colors.get(G.nodes[n]['reply_type'], '#B4B3B3') for n in nodes]
        node_sizes = []
        for n in nodes:
            post_type = G.nodes[n].get('post_type')
            if post_type == 'Original':
                size = 200
            elif post_type == 'Quote':
                size = 200
            else:
                size = 200 + 40 * G.nodes[n].get('thread_size', 1)
            node_sizes.append(size)

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=node_colors,
            node_size=node_sizes,
            node_shape=shape,
            edgecolors='black',
            linewidths=0.5
        )

    nx.draw_networkx_edges(G, pos, edge_color='#979595', arrows=True, arrowsize=12, width=1, alpha=0.4)

    labels = {n: G.nodes[n]['post_id'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=5)

    # Legends
    reply_legend = [Patch(color=color, label=label) for label, color in reply_type_colors.items()]
    shape_legend = [
        Line2D([0], [0], marker=shape, color='w', label=ptype, markerfacecolor='gray',
               markersize=10, markeredgecolor='black')
        for ptype, shape in post_type_shapes.items()
    ]

    plt.legend(handles=reply_legend + shape_legend, title="Legend", loc='upper right', fontsize=9)
    plt.title("Complete Thread Graph\nColored by Reply Type | Shaped by Post Type", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    G, df, uri_to_post_id = load_and_build_graph(FILE_PATH)

    # Visualize longest thread
    longest_path = find_longest_thread(G)
    draw_subgraph(G, longest_path, "Longest Thread")

    # Visualize all threads including missing parents
    draw_cross_engagement(G, df)

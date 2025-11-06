import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import csv
import spacy
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D

# Text preprocessing
def preprocess_text(file_path, min_words=20, max_words=150, min_final_words=10):
    nlp = spacy.load("en_core_web_sm")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    cleaned_paragraphs = []
    buffer = ""
    discarded_count = 0
    for para in raw_paragraphs:
        para = re.sub(r"\s+", " ", para)
        words = para.split()
        if len(words) < min_words:
            buffer += " " + para
            continue
        if buffer:
            if len(buffer.split()) >= min_final_words:
                cleaned_paragraphs.append(buffer.strip())
            else:
                discarded_count += 1
            buffer = ""
        if len(words) > max_words:
            doc = nlp(para)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            current_chunk = []
            word_count = 0
            for sent in sentences:
                words_in_sent = len(sent.split())
                if word_count + words_in_sent > max_words:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text.split()) >= min_final_words:
                        cleaned_paragraphs.append(chunk_text)
                    else:
                        discarded_count += 1
                    current_chunk = [sent]
                    word_count = words_in_sent
                else:
                    current_chunk.append(sent)
                    word_count += words_in_sent
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= min_final_words:
                    cleaned_paragraphs.append(chunk_text)
                else:
                    discarded_count += 1
        else:
            if len(words) >= min_final_words:
                cleaned_paragraphs.append(para.strip())
            else:
                discarded_count += 1
    if buffer and len(buffer.split()) >= min_final_words:
        cleaned_paragraphs.append(buffer.strip())
    elif buffer:
        discarded_count += 1
    print(f"Discarded {discarded_count} paragraphs with fewer than {min_final_words} words.")
    return cleaned_paragraphs

# Paragraph vectorization
def vectorize_paragraphs(paragraphs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(paragraphs, show_progress_bar=True)
    return np.array(embeddings)

# Figure saving
def save_figure(fig, filename, dpi=600):
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved as {filename}")

# UMAP scatter plot
def plot_umap_embeddings(embeddings, labels, title="Semantic Space Scatter Plot", save_path=None):
    reducer = umap.UMAP(random_state=42)
    reduced = reducer.fit_transform(embeddings)
    df_plot = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "label": labels})
    fig, ax = plt.subplots(figsize=(10,7))
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    sns.scatterplot(data=df_plot, x="x", y="y", hue="label",
                    palette={"Clarissa":"red", "Septimus":"blue"}, alpha=0.7, ax=ax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(frameon=False)
    if save_path:
        save_figure(fig, save_path)
    plt.show()
    return reduced

# HDBSCAN clustering plot
def plot_clusters(reduced, labels, save_path=None):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(reduced)
    df_plot = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "label": labels, "cluster": cluster_labels})
    fig, ax = plt.subplots(figsize=(10,7))
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    palette = sns.color_palette("tab20" if n_clusters <= 20 else "hls", n_colors=max(n_clusters, 1))
    cluster_colors = {}
    color_idx = 0
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_colors[cluster_id] = 'gray'
        else:
            cluster_colors[cluster_id] = palette[color_idx]
            color_idx += 1
    
    sns.scatterplot(data=df_plot, x="x", y="y", hue="cluster", style="label",
                    palette=cluster_colors, alpha=0.7, ax=ax)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.set_title("Semantic Space Clustering (HDBSCAN)")
    ax.legend(loc='center right', bbox_to_anchor=(-0.15, 0.5), frameon=False)
    if save_path:
        save_figure(fig, save_path)
    plt.show()
    return cluster_labels

# Semantic matching network
def semantic_matching_network(clarissa_emb, septimus_emb, clarissa_texts, septimus_texts, top_k=3, save_path=None):
    sim_matrix = cosine_similarity(clarissa_emb, septimus_emb)
    G = nx.Graph()
    for i in range(len(clarissa_texts)):
        G.add_node(f"C{i}", label="Clarissa")
    for j in range(len(septimus_texts)):
        G.add_node(f"S{j}", label="Septimus")
    for i in range(sim_matrix.shape[0]):
        top_idx = np.argsort(sim_matrix[i])[-top_k:]
        for j in top_idx:
            G.add_edge(f"C{i}", f"S{j}", weight=sim_matrix[i,j])
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12,8))
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n.startswith("C")],
                           node_color="red", label="Clarissa", alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n.startswith("S")],
                           node_color="blue", label="Septimus", alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    ax.set_title("Clarissa - Septimus Semantic Matching Network")
    ax.legend(markerscale=1.0, labelspacing=1.2, frameon=False)
    if save_path:
        save_figure(fig, save_path)
    plt.show()

# Cluster results saving
def save_cluster_results(paragraphs, labels, cluster_labels, output_file="cluster_results.csv"):
    df = pd.DataFrame({"paragraph": paragraphs, "label": labels, "cluster": cluster_labels})
    df.to_csv(output_file, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
    print(f"Clustering results saved to {output_file}")

# Matching results saving
def save_matching_results(clarissa_emb, septimus_emb, clarissa_paragraphs, septimus_paragraphs, top_k=3, output_file="semantic_matching.csv"):
    sim_matrix = cosine_similarity(clarissa_emb, septimus_emb)
    results = []
    for i in range(sim_matrix.shape[0]):
        top_idx = np.argsort(sim_matrix[i])[-top_k:][::-1]
        for j in top_idx:
            results.append({
                "clarissa_paragraph": clarissa_paragraphs[i],
                "septimus_paragraph": septimus_paragraphs[j],
                "similarity": sim_matrix[i,j]
            })
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
    print(f"Paragraph matching results saved to {output_file}")

# Combined visualization plot
def plot_combined_visualization(reduced, labels, cluster_labels,
                                clarissa_emb, septimus_emb,
                                clarissa_paragraphs, septimus_paragraphs,
                                top_k=2, title="Combined Semantic Space Visualization", save_path=None):
    fig, ax = plt.subplots(figsize=(12,9))
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    df_plot = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "label": labels, "cluster": cluster_labels})
    
    sns.scatterplot(data=df_plot, x="x", y="y", hue="label",
                    palette={"Clarissa":"red", "Septimus":"blue"},
                    alpha=0.7, s=50, ax=ax)
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
        cluster_points = df_plot[df_plot['cluster']==cluster_id][['x','y']].values
        center = cluster_points.mean(axis=0)
        radius = np.linalg.norm(cluster_points - center, axis=1).max()
        circle = plt.Circle(center, radius, color='gray', alpha=0.2, fill=True, label='_nolegend_')
        ax.add_patch(circle)
    
    sim_matrix = cosine_similarity(clarissa_emb, septimus_emb)
    for i in range(sim_matrix.shape[0]):
        top_idx = np.argsort(sim_matrix[i])[-top_k:]
        for j in top_idx:
            c_idx = i
            s_idx = len(clarissa_emb) + j
            ax.plot([reduced[c_idx,0], reduced[s_idx,0]],
                    [reduced[c_idx,1], reduced[s_idx,1]],
                    color='green', alpha=0.3, linewidth=0.7, label='_nolegend_')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Clarissa paragraph', markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Septimus paragraph', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], color='green', lw=2, label='Paragraph semantic matching (top_k)'),
        Line2D([0], [0], color='gray', lw=10, alpha=0.2, label='HDBSCAN cluster circles')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=False)
    ax.set_title(title)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    if save_path:
        save_figure(fig, save_path)
    plt.show()

# Top paragraphs selection
def select_top_paragraphs(sim_matrix, clarissa_paragraphs, septimus_paragraphs, 
                         clarissa_emb, septimus_emb, N=10, method='total_similarity'):
    if method == 'total_similarity':
        clarissa_scores = sim_matrix.sum(axis=1)
        septimus_scores = sim_matrix.sum(axis=0)
        top_c_indices = np.argsort(clarissa_scores)[-N:]
        top_s_indices = np.argsort(septimus_scores)[-N:]
    
    elif method == 'max_similarity':
        clarissa_scores = sim_matrix.max(axis=1)
        septimus_scores = sim_matrix.max(axis=0)
        top_c_indices = np.argsort(clarissa_scores)[-N:]
        top_s_indices = np.argsort(septimus_scores)[-N:]
    
    elif method == 'mutual_top':
        clarissa_scores = sim_matrix.max(axis=1)
        septimus_scores = sim_matrix.max(axis=0)
        top_c_indices = np.argsort(clarissa_scores)[-N:]
        top_s_indices = np.argsort(septimus_scores)[-N:]
    
    else:
        raise ValueError("Method must be 'total_similarity', 'max_similarity', or 'mutual_top'")
    
    top_clarissa = [clarissa_paragraphs[i] for i in top_c_indices]
    top_septimus = [septimus_paragraphs[j] for j in top_s_indices]
    
    return (clarissa_emb[top_c_indices], septimus_emb[top_s_indices], 
            top_clarissa, top_septimus, top_c_indices, top_s_indices)

# Main execution
if __name__ == "__main__":
    clarissa_file = "data/clarissa.txt"
    septimus_file = "data/septimus.txt"

    output_dir = "semantic_space_analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    TOP_K_MATCHING = 2
    
    clarissa_paragraphs = preprocess_text(clarissa_file)
    septimus_paragraphs = preprocess_text(septimus_file)
    
    print(f"Processing {len(clarissa_paragraphs)} Clarissa paragraphs and {len(septimus_paragraphs)} Septimus paragraphs")

    clarissa_emb = vectorize_paragraphs(clarissa_paragraphs)
    septimus_emb = vectorize_paragraphs(septimus_paragraphs)

    all_emb = np.vstack([clarissa_emb, septimus_emb])
    all_labels = ["Clarissa"]*len(clarissa_emb) + ["Septimus"]*len(septimus_emb)

    print("Generating full dataset visualizations...")
    reduced = plot_umap_embeddings(all_emb, all_labels, save_path=os.path.join(output_dir, "umap_scatter.tif"))
    cluster_labels = plot_clusters(reduced, all_labels, save_path=os.path.join(output_dir, "hdbscan_clusters.tif"))
    save_cluster_results(clarissa_paragraphs + septimus_paragraphs, all_labels, cluster_labels,
                         output_file=os.path.join(output_dir, "cluster_results.csv"))
    save_matching_results(clarissa_emb, septimus_emb, clarissa_paragraphs, septimus_paragraphs, top_k=TOP_K_MATCHING,
                          output_file=os.path.join(output_dir, "semantic_matching.csv"))
    semantic_matching_network(clarissa_emb, septimus_emb, clarissa_paragraphs, septimus_paragraphs, top_k=TOP_K_MATCHING,
                              save_path=os.path.join(output_dir, "matching_network.tif"))
    plot_combined_visualization(reduced, all_labels, cluster_labels,
                                clarissa_emb, septimus_emb,
                                clarissa_paragraphs, septimus_paragraphs,
                                top_k=TOP_K_MATCHING,
                                save_path=os.path.join(output_dir, "combined_visualization_with_legend.tif"))

    print("All analyses completed successfully!")
from functools import total_ordering
from heapq import nsmallest
from math import isnan
from os import altsep
from typing import Counter
from matplotlib import lines
from networkx.algorithms.bipartite.basic import color
import numpy as np
from numpy.core.fromnumeric import mean, size, sort
import scipy as sp
from scipy import sparse, linalg, stats
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from numpy.polynomial import Polynomial
import time
from scipy.sparse import csgraph
from scipy.sparse.csr import csr_matrix
import seaborn as sns
import math 
import pathlib


from ABmodel import generate_AB_graph, generate_AB_graph_ensure_m_edges

from memory_profiler import profile

# Complexity: O(E)
# We iterate through the edges once
def remove_self_edges(edgelist):
    orig_len = edgelist.shape[0]

    self_edge_idx = []

    for i in range(0, orig_len):
        if edgelist[i][0] == edgelist[i][1]:
            self_edge_idx.append(i)

    print(f"Self edge indices: {self_edge_idx}")

    edgelist = np.delete(edgelist, self_edge_idx, axis=0)

    print(f"{orig_len - edgelist.shape[0]} self edges removed")
    print(f"{edgelist.shape[0]} rows remain")

    return edgelist

def make_symmetric(edgelist):
    
    print(f"\nNumber edges before removing (i, j) (j, i) pairs {len(edgelist)}")

    # O(E) time, O(E) space

    edgelist = [frozenset(edge) for edge in edgelist if edge[0] != edge[1]]
    edgelist = set(edgelist)
    edgelist = list(edgelist)
    edgelist = np.array([list(edge) for edge in edgelist])

    print(f"Number edges after removing (i, j) (j, i) pairs {len(edgelist)}\n")

    return edgelist



def sanity_check(graph, details=False):

    A = graph
    A_minus_AT = A - A.T  

    print(f"\nIs A symmetric? (Is A - A.T all zeros?): {A_minus_AT.nnz == 0}")

    if details:
        A_minus_AT = A - A.T    

        A_plus_AT = A + A.T
        print(f"(A + A.T).nnz - (A - A.T).nnz: {A_plus_AT.nnz - A_minus_AT.nnz} (should give number of non-zero symmetric cells in A)")

        d = A_plus_AT.nnz - A_minus_AT.nnz
        print(f"A.nnz - ((A + A.T).nnz - (A - A.T).nnz): {A.nnz - d} (should give number of non-zero non-symmetric cells in A)")



def load_matrix(filename="./data/metabolic.edgelist.txt"):
    t1 = time.time()
    edgelist = np.loadtxt(filename, dtype=np.int32)

    edgelist = make_symmetric(edgelist)
    
    # Create the graph in CSC form
    rows = edgelist[:,0]
    cols = edgelist[:,1]
    data = np.ones(len(edgelist))

    # print(list(rows))

    n = np.amax(edgelist) + 1
    graph = sparse.coo_matrix((data ,(rows, cols)), shape=(n, n), dtype=np.int32)
    A = graph.tocsr()

    A = graph + graph.T

    print(f"A nnz: {A.nnz}")
    print(f"A shape: {A.shape}")
    degrees = get_degrees(A)
    total_degree = sum(degrees)
    print(f"\nTotal degree: {total_degree}")
    print(f"Average degree: {total_degree / A.shape[0]}")
    
    t2 = time.time()
    print(f"Time to load the graph: {t2-t1} s")

    return A



def a_plot_degree_distrib(A, model_name):

    degrees = get_degrees(A)

    total_degree = sum(degrees)
    print(f"A nnz: {A.nnz}")
    print(f"A shape: {A.shape}")
    print(f"\nTotal degree: {total_degree}")
    print(f"Average degree: {total_degree / A.shape[0]}")

    with open(f"./figs/{model_name}/model_stats.txt", "w") as outfile:
        outfile.write(f"A shape: {A.shape}\n")
        outfile.write(f"Total degree: {total_degree}\n")
        outfile.write(f"Average degree: {total_degree / A.shape[0]}\n")

    N = A.shape[0]

    # print(f"Degrees: {degrees}, len(degrees): {len(degrees)}")

    # Count the number of occurence of each degree
    counts = Counter(degrees)
    # print(counts)

    k, Nk = list(counts.keys()), np.array(list(counts.values()))

    # Calculate pk by dividing the number of nodes of a degree by total number of nodes
    pk = list(Nk / N)

    #clean up k and pk
    k_copy = k.copy()
    pk_copy = pk.copy()

    for i in range(0, len(k_copy)):
        if math.isnan(k_copy[i]) or k_copy[i] == 0:
            k.remove(k_copy[i])
            pk.remove(pk_copy[i])

    k = np.array(k)

    stop = np.ceil(np.log10(max(k)))

    # TODO: Change the num_bins on a per graph basis
    # This is a dark art, too many bins and the first few contain no nodes 
    num_bins = int(stop) * 7
    
    # print(f"Num bins: {num_bins}")
    bins = np.logspace(start=0, stop=stop, base=10, num=num_bins, endpoint=True)

    # bins = np.linspace(start=min(k), stop=max(k), num=num_bins)

    # print(f"Bins: {bins}")

    # Subtract 1 to get 0-based indices
    digitized = np.digitize(degrees, bins)

    # print(f"digitized: {digitized}")

    bin_counts = Counter(digitized)
    # print(f"Nn_count: {bin_counts}")

    Nn = []

    for i in range(0, num_bins):
        count = bin_counts[i]
        # print(f"There are {count} nodes in bin {i}")
        Nn.append(count)


    while len(Nn) < (num_bins - 1):
        Nn.append(0)

    bn = np.array(bins[1:] - bins[:-1])

    bn = np.append(bn, [bn[-1]*2])


    # THIS IS WEIRD
    # From NS Book Section 4.12.3. When using log binning, p<kn> is given by 
    # Nn/bn : (Number of nodes in bin n) / (size of bin n)
    # However, dividing only by the size of the bin can give values > 0. 
    # E.g. For a bin of size=2 capturing nodes of degree 1 and 2 if there are > 2 nodes in it...
    #
    # NI book Section 10.4.1 does not give an explicit formula but says "We have to be careful to 
    # normalize each bin by its width". Thus it seems we have to still divide by N, while also 
    # correcting for bin size.
    pkn = np.array(Nn) / bn
    pkn = list(pkn / N)


    # Average degree of nodes in each bin
    kn = [degrees[digitized == i].mean() for i in range(0, len(bins))]



    #clean up kn and pkn
    kn_copy = kn.copy()
    pkn_copy = pkn.copy()

    for i in range(0, len(kn_copy)):
        if math.isnan(kn_copy[i]) or kn_copy[i] == 0:
            kn.remove(kn_copy[i])
            pkn.remove(pkn_copy[i])

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))
    # fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

    ax1.set_title("Raw Data")
    ax1.set_ylabel("P(k)")
    ax1.set_xlabel("k")
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
    ax1.tick_params(which='both', direction="in")
    ax1.grid(True, which='both', linestyle='--')
    ax1.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    ax1.scatter(k, pk)

    ax2.set_title("Log Binned Data")
    ax2.set_ylabel("P(k)")
    ax2.set_xlabel("k")
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=10)
    ax2.tick_params(which='both', direction="in")
    ax2.grid(True, which='both', linestyle='--')
    ax2.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    ax2.scatter(kn, pkn)

    fit1 = Polynomial.fit(np.log10(k), np.log10(pk), deg=1)
    print(f"Gamma (unbinned): {fit1.coef[0]}")

    yfit1 = lambda x: np.power(10, fit1(np.log10(x)))
    ax1.plot(k, yfit1(k), c='r', label=f"Exp: {fit1.coef[0]}")
    ax1.legend()

    fit2 = Polynomial.fit(np.log10(kn), np.log10(pkn), deg=1)
    print(f"Gamma (log binned): {fit2.coef[0]}")

    yfit2 = lambda x: np.power(10, fit2(np.log10(x)))
    ax2.plot(kn, yfit2(kn), c='r', label=f"Exp: {fit2.coef[0]}")
    ax2.legend()

    fig.suptitle(f'{model_name}: Degree Distribution')
    fig.tight_layout()

    plt.savefig(f"./figs/{model_name}/1 - Degree Distrib.png", format="png")
    plt.close()


def get_degrees(graph):
    _, diag = sparse.csgraph.laplacian(graph, return_diag=True)
    return diag


def get_clustering_coefs(graph, avg=False):

    # TODO Remove this, just to check our results
    import networkx as nx 

    G = nx.from_scipy_sparse_matrix(graph)
    print(f"NX avg clustering of graph: {nx.average_clustering(G)}")


    t1 = time.time()
    
    A3 = graph ** 3

    deg = get_degrees(graph)

    N = A3.shape[0]

    A3_diag = A3.diagonal()

    clustering_coefs = [A3_diag[i] / (deg[i] * (deg[i] - 1)) if (deg[i] - 1) > 0 else 0 for i in range(N)]

    clustering_coefs = np.array(clustering_coefs, dtype=np.float32)

    print(f"My avg clustering: {sum(clustering_coefs) / N}")

    t2 = time.time()

    print(f"Time to get clustering coefs: {t2-t1} s")

    if avg:
        return clustering_coefs, deg, sum(clustering_coefs) / N

    return clustering_coefs, deg


def b_plot_clustering_coef_distrib(A, model_name):
    cc, _, avg_cc = get_clustering_coefs(A, avg=True)

    print(f"\nAvg cc: {avg_cc}")

    counts = Counter(cc)

    fig, ax = plt.subplots(figsize=(13, 7))


    ax.set_title(f"{model_name}: Clustering Coefficient Distribution")
    ax.set_xlabel("Local Clustering Coefficient")
    ax.set_ylabel("Density")

    avg_cc_label = "{:.3f}".format(avg_cc)

    # Draw line for the mean
    ax.axvline(x=avg_cc, color='grey', linestyle='--', alpha=.7)

    # Used to position the mean label
    trans = ax.get_xaxis_transform()
    ax.text(avg_cc + .02, .9, f"Mean = {avg_cc_label}", transform=trans)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    ax.set_xlim(0,1)

    # ax.hist(cc, bins=bins, color='purple')
    # sns.displot(cc, kind='kde', kde_kws = {'color': 'purple', 'shade': True}, ax=ax)
    sns.kdeplot(data=cc, fill=True, color='purple', ax=ax)

    fig.tight_layout()
    plt.savefig(f"./figs/{model_name}/2 - Clustering Coef.png", format="png")
    plt.close()

# It does phonecalls in ~7min
# About O(n3)
def c_plot_shortest_paths(graph, model_name):

    N = graph.shape[0]

    # # TODO Remove this, its just to check our results
    # import networkx as nx 

    # G = nx.from_scipy_sparse_matrix(graph)
    # nx_sps = nx.shortest_path_length(G)


    # mean_distances = []
    # for _, sps in nx_sps:
    #     mean_dist = sum(sps.values()) / len(sps.values())
    #     mean_distances.append(mean_dist)

    # mean_distance_global = np.mean(mean_distances)
    # print(f"NX Average path length: {mean_distance_global}")


    t1 = time.time()

    # Saw the note in documentation about D possible conflict with directed=False 
    # but our graphs are always symmetric so should work fine.
    # shortest_paths = sparse.csgraph.shortest_path(graph, method='D', directed=False, unweighted=True )
    shortest_paths = sparse.csgraph.shortest_path(graph, method='D', directed=False)
    shortest_paths = shortest_paths[np.isfinite(shortest_paths)]

    # As in the NI book Section 10.2
    avg_sp = shortest_paths.sum() / (N ** 2)
    
    print(f"My Average path length: {avg_sp}")

    shortest_paths = csr_matrix(shortest_paths)

    t2 = time.time()
    print(f"Time to get shortest paths: {t2-t1} s")

    fig, ax = plt.subplots(figsize=(9,6))

    trans = ax.get_xaxis_transform()

    ax.set_title(f"{model_name}: Shortest Path Distribution")
    ax.set_xlabel("Shortest Path Length")
    ax.set_ylabel("Count")

    avg_cc_label = "{:.2f}".format(avg_sp)
    ax.axvline(x=avg_sp, color='grey', linestyle='--', alpha=.7)
    ax.text(avg_sp + .02, .9, f"Mean = {avg_cc_label}", transform=trans)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)


    ax.hist(shortest_paths.data, color='purple', bins=5)
    # sns.distplot(shortest_paths.data, hist = False, kde = True, kde_kws = {'color': 'purple', 'shade': True}, ax=ax)

    fig.tight_layout()
    plt.savefig(f"./figs/{model_name}/3 - Shortest Paths.png", format="png")
    plt.close()




def d_get_connected_compo(graph, model_name):
    t1 = time.time()
    con_comp, labels = sparse.csgraph.connected_components(graph, directed=False, return_labels=True)
    print(f"\nNumber of Connected Components: {con_comp}")

    N = labels.shape[0]

    counts = Counter(labels)

    GCC, GCC_len = counts.most_common(1)[0]

    nodes_in_GCC = GCC_len / N

    print(f"Proportion of nodes in GCC: {nodes_in_GCC}")

    t2 = time.time()    
    print(f"Time to get comnnected components: {t2-t1} s")

    with open(f"./figs/{model_name}/4 - Connected Components.txt", "w") as outfile:
        outfile.write(f"Number of Connected Components:\t\t{con_comp}\n")
        outfile.write(f"Proportion of nodes in GCC:\t\t{nodes_in_GCC}\n")


def e_get_eigenval_distrib(graph, model_name):

    t1 = time.time()

    L = sparse.csgraph.laplacian(graph)

    # k = graph.shape[0]-1
    k = 100

    # Need to use float
    eigenvals_L = sparse.linalg.eigsh(L.asfptype(), k=k, return_eigenvectors=False, which = 'SM')

    t2 = time.time()    
    print(f"Time to gt {k} Eigenvals: {t2-t1} s")

    num_zeros = sparse.csgraph.connected_components(sp.sparse.csgraph.laplacian(graph), directed=False, return_labels=False)

    spectral_gap = 0

    if num_zeros < len(eigenvals_L):
        spectral_gap = eigenvals_L[len(eigenvals_L)-1-num_zeros]
        print(f"Spectral gap: {spectral_gap}")
    else: 
        print("Spectral gap: 0")
  
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.set_title(f"{model_name}: Distribution of {k} smallest eigenvalues of Laplacian matrix")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")   

    # Create empty plot with blank marker containing the extra label
    # Format in scientific notation
    spec = "{:e}".format(spectral_gap)
    ax.plot([], [], ' ', label=f"Spectral gap: {spec}")

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    # sns.kdeplot(data=eigenvals_L.real, fill=True, color='purple', ax=ax, bw_adjust=0.2)

    ax.hist(eigenvals_L.real, color='purple', bins=20)

    ax.legend()
    fig.tight_layout()

    plt.savefig(f"./figs/{model_name}/5 - Eigenvalues.png", format="png")
    plt.close()



def f_get_degree_correl(graph, model_name):
    t1 = time.time()
    degs = get_degrees(graph)
    n = degs.shape[0]

    X = []
    Y = []

    # O(n2)
    # Efficient row slicing with CSR
    for i in range(n):
        if i % 2000 == 0:
            print(f"on row {i}")
        row = graph.getrow(i).toarray()[0]
        # print(f"Row {i}: {row}")
        
        for j in range(n):
            if row[j] == 1:
                # print(f"Node {i} links to node {j}")
                X.append(degs[i])
                Y.append(degs[j])



    t2 = time.time()    
    print(f"Time to calculate degree correlations: {t2-t1} s")

    # Doesn't seem like it should work but compared with the one NX gives, it seems fine
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print(f"R value {r_value}")

    import networkx

    G = networkx.from_scipy_sparse_matrix(graph)
    r_value = networkx.algorithms.assortativity.degree_pearson_correlation_coefficient(G)
    print(f"R value from nx {r_value}")


    fig, ax = plt.subplots(figsize=(9, 6))

    ax.set_title(f"{model_name}: Degree Correlation between nodes i and j")
    ax.set_xlabel("degree of node i")
    ax.set_ylabel("degree of node j")

    # Create empty plot with blank marker containing the extra label
    R = "{:.2f}".format(r_value)
    ax.plot([], [], ' ', label=f"R: {R}")

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    # Get the point density
    c = Counter(zip(X,Y))
    density = [100 * c[(xx,yy)] for xx,yy in zip(X,Y)]

    ax.scatter(X, Y, s=1, c=density, cmap='gist_yarg', norm=colors.LogNorm(vmin=1, vmax=max(density)/1.5))

    # Old version, showed less detail
    # sns.histplot(data=dataset, x="di", y="dj", discrete=(True, True), binwidth=50, cbar=True, ax=ax)

    ax.legend() 
    fig.tight_layout()
    plt.savefig(f"./figs/{model_name}/6 - Degree Correlation.png", format="png")
    plt.close()



def g_plot_clustering_degree_rel(A, model_name):

    cc, d = get_clustering_coefs(A)

    print(f"Size of CC: {len(cc)}")
    print(f"Size of degrees: {len(d)}")

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.set_title(f"{model_name}: Clustering Coefficient vs Degree")
    ax.set_xlabel("Local Clustering Coefficient")
    ax.set_ylabel("Degree")
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    # Get the point density to change color based on counts
    c = Counter(zip(cc,d))
    density = [100 * c[(xx,yy)] for xx,yy in zip(cc,d)]

    ax.scatter(cc, d, c=density, cmap='gist_yarg', norm=colors.LogNorm(vmin=1, vmax=max(density)/1.5))

    fig.tight_layout()
    plt.savefig(f"./figs/{model_name}/7 - CC vs Degree.png", format="png")
    plt.close()


def main():
    t1 = time.time()

    print("\n\n")

    # G = load_matrix("./data/powergrid.edgelist.txt")
    # G = load_matrix("./data/collaboration.edgelist.txt")
    # G = load_matrix("./data/email.edgelist.txt")
    # G = load_matrix("./data/AB_ensure_n1039_m5.edgelist.txt")

    G = generate_AB_graph_ensure_m_edges(4941, 1)

    # G = load_matrix("./data/AB_ensure_n2018_m2.edgelist.txt")

    model_name = "BA model n=4941 m=1"

    pathlib.Path(f"./figs/{model_name}").mkdir(parents=True, exist_ok=True)

    a_plot_degree_distrib(G, model_name)

    b_plot_clustering_coef_distrib(G, model_name)

    c_plot_shortest_paths(G, model_name)

    d_get_connected_compo(G, model_name)

    e_get_eigenval_distrib(G, model_name)

    f_get_degree_correl(G, model_name)

    g_plot_clustering_degree_rel(G, model_name)


    t2 = time.time()
    print(f"Total running time: {t2-t1} s")

    print("\n\n\n")


if __name__=='__main__': 
    main()

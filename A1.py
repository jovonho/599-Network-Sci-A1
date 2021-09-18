from functools import total_ordering
from os import altsep
from typing import Counter
from matplotlib import lines
import numpy as np
from numpy.core.fromnumeric import size
import scipy as sp
from scipy import sparse, linalg, stats
import pandas as pd
import matplotlib.pyplot as plt 
from numpy.polynomial import Polynomial
import time
import seaborn as sns

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

    # O(E) time, O(E^2) space
    # edgelist = list(set(frozenset(edge) for edge in edgelist if edge[0] != edge[1]))
    edgelist = [frozenset(edge) for edge in edgelist if edge[0] != edge[1]]

    edgelist = set(edgelist)

    edgelist = list(edgelist)

    # Complexity?
    # This will remove one of each symmetric edge of form (i, j), (j, i)
    # edgelist = list(set(edgelist))
    # print(edgelist)

    edgelist = np.array([list(edge) for edge in edgelist])

    print(f"Number edges after removing (i, j) (j, i) pairs {len(edgelist)}")

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


def make_symmetric2(graph):
    A = graph.toarray()
    A = np.maximum(A, A.transpose())
    return sparse.csr_matrix(A)


def load_matrix(filename="./data/metabolic.edgelist.txt"):
    t1 = time.time()
    edgelist = np.loadtxt(filename, dtype=np.int32)

    # edgelist = remove_self_edges(edgelist)
    edgelist = make_symmetric(edgelist)
    
    # Create the graph in CSC form
    rows = edgelist[:,0]
    cols = edgelist[:,1]
    data = np.ones(len(edgelist))

    n = np.amax(edgelist) + 1
    graph = sparse.coo_matrix((data ,(rows, cols)), shape=(n, n), dtype=np.int32)
    graph = graph.tocsr()

    # TODO: Complexity?
    A = graph + graph.T


    print(f"A nnz: {A.nnz}")
    print(f"A shape: {A.shape}")
    
    t2 = time.time()
    print(f"Time to load the graph: {t2-t1} s")

    # sanity_check(A)

    return A



def plot_degree_distrib(A):

    degrees = np.array([sum(line) for line in A.A])
    total_degree = sum(degrees)

    print(f"Degrees: {degrees}, len(degrees): {len(degrees)}")
    print(f"Total degree: {total_degree}")


    counts = Counter(degrees)
    # print(f"counts: {counts}")
    print(f"counts[5]: {counts[5]}")

    X, Y = list(counts.keys()), list(counts.values())

    print(Y)
    print(f"{X[4]}: {Y[4]}")

    Y = Y / total_degree

    print(Y)

    print(f"{X[4]}: {Y[4]}")
    

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))

    ax1.set_title("Raw Data")
    ax1.set_ylabel("P(k)")
    ax1.set_xlabel("k")
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
    ax1.tick_params(which='both', direction="in")

    ax1.scatter(X, Y)

    # plt.show()

    # fit = Polynomial.fit(np.log10(X), np.log10(Y), deg=1)
    # print(f"Exponent: {fit.coef[0]}")

    # yfit = lambda x: np.power(10, fit(np.log10(x)))
    # ax1.plot(X, yfit(X))


    # Log binning
    bins = np.logspace(np.log10(min(X)), np.log10(max(X)))
    print(f"Bins: {bins}\n")

    widths = (bins[1:] - bins[:-1])
    print(f"Widths: {widths}\n")

    # Calculate histogram
    hist = np.histogram(X, bins=bins, weights=Y)

    x_hist = np.histogram(X, bins )
    y_hist = np.histogram(X, bins)

    print(f"X hist: {x_hist[0]}")
    print(f"Y hist: {y_hist[0]}")  

    x_hist_with_weights = np.histogram(X, bins, weights=X)
    y_hist_with_weights = np.histogram(X, bins, weights=Y)

    print(f"X hist w weights: {x_hist_with_weights[0]}")
    print(f"Y hist w weights: {y_hist_with_weights[0]}")


    x_bin_means = np.histogram(X, bins, weights=X)[0] / np.histogram(X, bins)[0]
    y_bin_means = np.histogram(X, bins, weights=Y)[0] / np.histogram(X, bins)[0]

    # print(f"Hist[0]:\n{hist[0]}\n")
    print(f"x_bin_means:\n{x_bin_means}\n")
    print(f"y_bin_means:\n{y_bin_means}\n")

    # normalize by bin width
    hist_norm = hist[0]/widths

    # print(f"\nhist_norm:\n{hist_norm}")

    ax2.set_title("Log Binned")
    ax2.set_ylabel("P(k)")
    ax2.set_xlabel("k")
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=10)
    ax2.tick_params(which='both', direction="in")

    # ax2.scatter(x_bin_means, y_bin_means, c='r')
    ax2.scatter(hist_norm, widths, c='r')
    # ax2.plot(X, yfit(X))

    plt.show()


def get_degrees(graph):
    _, diag = sparse.csgraph.laplacian(graph, return_diag=True)
    return diag


def get_clustering_coefs(graph, avg=False):
    t1 = time.time()
    
    A3 = graph ** 3

    deg = get_degrees(graph)

    n = A3.shape[0]

    #Note: accessing by index is slow

    clustering_coefs = [A3[i,i] / (deg[i] * (deg[i] - 1)) if (deg[i] - 1) > 0 else 0 for i in range(n)]

    A3_diag = A3.diagonal()
    print(A3_diag)

    clustering_coefs2 = [A3_diag[i] / (deg[i] * (deg[i] - 1)) if (deg[i] - 1) > 0 else 0 for i in range(n)]

    print(f"If method2 == method1? {clustering_coefs == clustering_coefs2}")

    # print(f"clustering coefs: {clustering_coefs}")
    t2 = time.time()

    print(f"Time to get clustering coefs: {t2-t1} s")

    if avg:
        return clustering_coefs, deg, sum(clustering_coefs) / sum(deg)

    return clustering_coefs, deg
    

# It does phonecalls in ~7min
# About O(n3)
def plot_shortest_paths(graph):
    t1 = time.time()
    # Saw the note in documentation about D possible conflict with directed=False 
    # but out graphs are always symmetric so should work fine.
    # shortest_paths = sparse.csgraph.shortest_path(graph, method='D', directed=False, unweighted=True )

    # TODO Find out which one is fastest
    shortest_paths = sparse.csgraph.shortest_path(graph, method='D', directed=False)

    # Can give memory error with large graphs
    shortest_paths = sparse.triu(shortest_paths)

    avg_sp = shortest_paths.sum() / shortest_paths.nnz

    print(f"Average path length: {avg_sp}")

    counts = Counter(shortest_paths.data)

    print(counts)

    print(f"List of all shortest path lengths: {shortest_paths.data}")

    t2 = time.time()
    print(f"Time to get shortest paths: {t2-t1} s")

    fig, ax = plt.subplots()

    trans = ax.get_xaxis_transform()

    ax.set_title("Shortest Path Distribution")
    ax.set_xlabel("Shortest Path Length")
    ax.set_ylabel("Count")

    avg_cc_label = "{:.2f}".format(avg_sp)
    ax.axvline(x=avg_sp, color='grey', linestyle='--', alpha=.7)
    ax.text(avg_sp + .02, .9, f"Mean = {avg_cc_label}", transform=trans)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)


    ax.hist(shortest_paths.data, color='purple')
    # sns.distplot(shortest_paths.data, hist = False, kde = True, kde_kws = {'color': 'purple', 'shade': True}, ax=ax)

    fig.tight_layout()

    plt.show()





def get_connected_compo(graph):
    t1 = time.time()
    con_comp, labels = sparse.csgraph.connected_components(graph, directed=False, return_labels=True)
    print(f"Number of Connected Components: {con_comp}")

    N = labels.shape[0]
    print(f"Length of labels (num nodes): {N}")

    counts = Counter(labels)

    GCC, GCC_len = counts.most_common(1)[0]

    nodes_in_GCC = GCC_len / N

    print(f"Proportion of nodes in GCC: {nodes_in_GCC}")

    t2 = time.time()    
    print(f"Time to get comnnected components: {t2-t1} s")




def get_degree_correl(graph):
    t1 = time.time()
    degs = get_degrees(graph)
    n = degs.shape[0]

    X = []
    Y = []

    # O(n2)
    # Efficient row slicing with CSR
    for i in range(n):
        row = graph[i,:].toarray()[0]
        # print(f"Row {i}: {row}")
        
        for j in range(n):
            if row[j] == 1:
                # print(f"Node {i} links to node {j}")
                X.append(degs[i])
                Y.append(degs[j])


    # O(n2)
    # Efficient row slicing with CSR
    for i in range(n):
        row = graph.getrow(i).toarray()[0]
        # print(f"Row {i}: {row}")
        
        for j in range(n):
            if row[j] == 1:
                # print(f"Node {i} links to node {j}")
                X.append(degs[i])
                Y.append(degs[j])


    dataset = pd.DataFrame({'di': X, 'dj': Y}, columns=['di', 'dj'])

    t2 = time.time()    
    print(f"Time to calculate degree correlations: {t2-t1} s")

    # correl = np.corrcoef(X, Y)
    # print(f"Correl {correl}")

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print(f"R2 {r_value}")
    print(f"slope {slope}")
    print(f"intercept {intercept}")

    # m, b = np.polyfit(X, Y, 1)
    # print(f"m {m}")
    # print(f"b {b}")

    fig, ax = plt.subplots()
    ax.set_title("Degree Correlation")
    ax.set_xlabel("di")

    ax.set_ylabel("dj")

    planets = sns.load_dataset('planets')

    print(planets)

    # Should be length 2L * 2
    print(dataset.shape)
    print(dataset)

    print(f"Max X: {max(X)}")
    print(f"Max Y: {max(Y)}")


    # Create empty plot with blank marker containing the extra label
    R = "{:.2f}".format(r_value)
    ax.plot([], [], ' ', label=f"R: {R}")

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    # ax.scatter(X, Y, color='purple', s=1 )
    sns.histplot(data=dataset, x="di", y="dj", discrete=(True, True), cbar=True)

    ax.legend() 
    fig.tight_layout()

    plt.show()




def b_plot_clustering_coef_distrib(A):
    cc, _, avg_cc = get_clustering_coefs(A, avg=True)

    print(f"Avg cc: {avg_cc}")

    counts = Counter(cc)

    X, Y = list(counts.keys()), list(counts.values())

    bins = np.linspace(min(X), max(X), num=100)

    y_bins = np.linspace(min(Y), max(Y), num=10)

    fig, ax = plt.subplots(figsize=(13, 7))



    ax.set_title("Metabolic: Clustering Coef Distribution")
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

    # ax.hist(cc, bins=bins, color='purple')
    # sns.displot(cc, kind='kde', kde_kws = {'color': 'purple', 'shade': True}, ax=ax)
    sns.kdeplot(data=cc, fill=True, color='purple', ax=ax)

    fig.tight_layout()
    plt.show()


def eigenval_distrib(graph):

    t1 = time.time()

    L = sparse.csgraph.laplacian(graph)

    # Need to use float
    eigenvals = sparse.linalg.eigsh(graph.asfptype(), k=graph.shape[0]-1, return_eigenvectors=False)
    eigenvals_L = sparse.linalg.eigsh(L.asfptype(), k=L.shape[0]-1, return_eigenvectors=False)

    t2 = time.time()    
    print(f"Time to gt Eigenvals: {t2-t1} s")

    print(f"eigenvals shape: {eigenvals.shape}")
    print(f"Eigenvals: {eigenvals}")

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13, 7))

    ax1.set_title("Eigenvalue Distribution of Adjacency Matrix")
    ax1.set_xlabel("Eigenvalue")
    ax1.set_ylabel("Density")

    ax1.grid(True, which='both', linestyle='--')
    ax1.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    sns.kdeplot(data=eigenvals, fill=True, color='purple', ax=ax1)

    ax2.set_title("Eigenvalue Distribution of Laplacian Matrix")
    ax2.set_xlabel("Eigenvalue")
    ax2.set_ylabel("Density")

    ax2.grid(True, which='both', linestyle='--')
    ax2.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    sns.kdeplot(data=eigenvals_L, fill=True, color='purple', ax=ax2)

    fig.tight_layout()
    plt.show()


# Complexity: 
def g_plot_clustering_degree_rel(A):

    cc, d = get_clustering_coefs(A)

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.set_title("Clustering Coef vs Degree")
    ax.set_xlabel("Local Clustering Coefficient")
    ax.set_ylabel("Degree")
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    # ax.scatter(cc, d, c='purple')

    dataset = pd.DataFrame({'clust_coef': cc, 'degree': d}, columns=['clust_coef', 'degree'])

    sns.histplot(data=dataset, x="di", y="dj", discrete=(True, True), cbar=True)

    fig.tight_layout()

    plt.show()



def main():
    t1 = time.time()

    print("\n\n")


    # A = load_matrix("./data/phonecalls.edgelist.txt")
    A = load_matrix("./data/powergrid.edgelist.txt")
    # A = load_matrix("./data/test.txt")
    # print(A.todense())

    plot_degree_distrib(A)

    # plot_shortest_paths(A)

    # get_connected_compo(A)

    # get_degree_correl(A)

    # eigenval_distrib(A)

    # b_plot_clustering_coef_distrib(A)

    # g_plot_clustering_degree_rel(A)

    t2 = time.time()
    print(f"Total running time: {t2-t1} s")


    print("\n\n\n")


if __name__=='__main__': 
    main()
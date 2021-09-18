from functools import total_ordering
from os import altsep
from typing import Counter
from matplotlib import lines
import numpy as np
import scipy as sp
from scipy import sparse, linalg
import pandas as pd
import matplotlib.pyplot as plt 
from numpy.polynomial import Polynomial
import time

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

    fit = Polynomial.fit(np.log10(X), np.log10(Y), deg=1)
    print(f"Exponent: {fit.coef[0]}")

    yfit = lambda x: np.power(10, fit(np.log10(x)))
    ax1.plot(X, yfit(X))


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
    ax2.plot(X, yfit(X))

    plt.show()


def get_degrees(graph):
    _, diag = sparse.csgraph.laplacian(graph, return_diag=True)
    return diag


def get_clustering_coefs(graph, avg=False):
    t1 = time.time()
    
    A3 = graph ** 3

    deg = get_degrees(graph)

    n = A3.shape[0]

    clustering_coefs = [A3[i,i] / (deg[i] * (deg[i] - 1)) if (deg[i] - 1) > 0 else 0 for i in range(n)]

    # print(f"clustering coefs: {clustering_coefs}")
    t2 = time.time()

    print(f"Time to get clustering coefs: {t2-t1} s")

    if avg:
        return clustering_coefs, deg, sum(clustering_coefs) / sum(deg)

    return clustering_coefs, deg
    

def plot_shortest_paths(graph):
    # Saw the note in docu, but out graphs are always symmetric so should work fine.
    shortest_paths = sparse.csgraph.shortest_path(graph, method='D', directed=False)




def b_plot_clustering_coef_distrib(A):
    cc, _, avg_cc = get_clustering_coefs(A, avg=True)

    # print(f"Avg cc: {avg_cc}")

    counts = Counter(cc)

    X, Y = list(counts.keys()), list(counts.values())

    bins = np.linspace(min(X), max(X), num=100)

    y_bins = np.linspace(min(Y), max(Y), num=10)

    fig, ax = plt.subplots(figsize=(13, 7))

    # the x coords of this transformation are data, and the
    # y coord are axes
    trans = ax.get_xaxis_transform()


    ax.set_title("Clustering Coef Distribution")
    ax.set_xlabel("Local Clustering Coefficient")
    ax.set_ylabel("Count")

    avg_cc_label = "{:.2f}".format(avg_cc)

    ax.axvline(x=avg_cc, color='grey', linestyle='--', alpha=.7)
    ax.text(avg_cc + .02, .9, f"Mean = {avg_cc_label}", transform=trans)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    ax.hist(cc, bins=bins, color='purple')
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
    ax.scatter(cc, d, c='purple')
    fig.tight_layout()

    plt.show()

def main():
    t1 = time.time()

    print("\n\n")


    # A = load_matrix("./data/email.edgelist.txt")
    A = load_matrix("./data/phonecalls.edgelist.txt")
    # A = load_matrix("./data/test.txt")
    # print(A.todense())

    # plot_degree_distrib(A)
    # cc, d, avg_cc = get_clustering_coefs(A, avg=True)

    # plot_shortest_paths(A)

    b_plot_clustering_coef_distrib(A)

    # g_plot_clustering_degree_rel(A)

    t2 = time.time()
    print(f"Total running time: {t2-t1} s")


    print("\n\n\n")


if __name__=='__main__': 
    main()
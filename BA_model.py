import numpy as np
from scipy.sparse import coo_matrix
import time
import logging
from logging import handlers
import sys


log = logging.getLogger(__name__)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

fh = handlers.RotatingFileHandler(filename="./log/BA_model.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)

log.addHandler(ch)
log.addHandler(fh)


# Complexity: O(n^2)
def generate_BA_graph_ensure_m_edges(n, m):
    t1 = time.time()

    # Initialize our data structures

    # These list are in COO format, where an Matrix(row[i], col[i]) = 1 if there is a link
    # We want our graph to be undirected so we insert edges in both directions.
    row = [0, 1]
    col = [1, 0]

    # This dictionary will keep track of the degrees of each nodes
    nodes = {0: 1, 1: 1}

    total_degree = 2

    print(f"*** Generating BA Model with n: {n}, m: {m} ***")

    for i in range(2, n):

        m2 = m

        if i % 500 == 0:
            print(f"Adding node {i}")

        # Copy the current nodes and their degree
        candidate_nodes = nodes.copy()
        total_degree_at_t = total_degree
        links_made = 0

        log.debug(f"Adding new node {i}")
        log.debug(f"\tDegrees at time t={i}: {candidate_nodes}")

        # TO not have infinite lop when m > 2
        if i <= m:
            m2 = i

        while links_made < m2:
            log.debug(f"linking node {i}")
            # print(f"linking node {i}, need to make {m2} edges")

            # Here we iterate over the existing nodes always in the same order
            for k, k_deg in candidate_nodes.items():

                log.debug(f"\tTrying to link node {k}")

                # Compute the probability of linking node k to i
                p = k_deg / total_degree_at_t
                log.debug(f"\tProb of linking node {k} to {i}: {p}")

                make_link = np.random.choice([0, 1], p=[(1 - p), p])

                if make_link:
                    links_made += 1

                    # TODO: Keep track of which nodes we've connected to, maybe it makes multiple edges to a single node!
                    log.debug(f"\t*** Nodes {k} and {i} linked!")

                    # Add the new link to the row and col lists
                    # We add the link in both directions (k, i) and (i, k) so the matrix will be symmetric
                    row.extend([k, i])
                    col.extend([i, k])

                    # Increment the degrees of each node in the dictionary to use it when adding the next node
                    if i in nodes:
                        nodes[i] += 1
                    else:
                        nodes[i] = 1

                    nodes[k] += 1

                    # Total degree increases by 2, since we add symmetric links
                    total_degree += 2

                    if links_made == m:
                        log.debug(f"\t******All {m2} links made for node {i}!!")
                        break

            log.debug(f"We've iterated through all the nodes. links made: {links_made} vs {m2}")

    # Data will be all ones
    nnz = len(row)
    data = np.ones(nnz)

    # Generate the sparse matrix only at the end, from the coordinate lists
    graph = coo_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)

    t2 = time.time()
    print(f"Time to generate AB model(n={n}, m={m}): {t2-t1} s")

    return graph.tocsr()


# We tweak the BA model to prefenentially attach to nodes with low degree.
# We just do 1 - p of the orginal model, so we invert the probability of attaching.
def generate_BA_graph_preferential_inattachment(n, m):
    t1 = time.time()

    # Initialize our data structures

    # These list are in COO format, where an Matrix(row[i], col[i]) = 1 if there is a link
    # We want our graph to be undirected so we insert edges in both directions.
    row = [0, 1]
    col = [1, 0]

    # This dictionary will keep track of the degrees of each nodes
    nodes = {0: 1, 1: 1}

    total_degree = 2

    print(f"*** Generating BA Model with n: {n}, m: {m} ***")

    for i in range(2, n):

        if i % 1000 == 0:
            print(f"Adding node {i}")

        # Copy the current nodes and their degree
        candidate_nodes = nodes.copy()
        # Must get the keys te be able to remove values from it while iterating
        candidate_node_keys = list(candidate_nodes.keys())
        total_degree_at_t = total_degree
        links_made = 0

        log.debug(f"Adding new node {i}")
        log.debug(f"\tDegrees at time t={i}: {candidate_nodes}")

        while links_made < m:
            log.debug(f"linking node {i}")

            # Here we iterate over the existing nodes always in the same order
            for k, k_deg in candidate_nodes.items():

                log.debug(f"\tTrying to link node {k}")

                # Compute the probability of linking node k to i
                p = 1 - (k_deg / total_degree_at_t)
                log.debug(f"\tProb of linking node {k} to {i}: {p}")

                make_link = np.random.choice([0, 1], p=[(1 - p), p])

                if make_link:
                    links_made += 1

                    log.debug(f"\t*** Nodes {k} and {i} linked!")

                    # Add the new link to the row and col lists
                    # We add the link in both directions (k, i) and (i, k) so the matrix will be symmetric
                    row.extend([k, i])
                    col.extend([i, k])

                    # Increment the degrees of each node in the dictionary to use it when adding the next node
                    if i in nodes:
                        nodes[i] += 1
                    else:
                        nodes[i] = 1

                    nodes[k] += 1

                    # Total degree increases by 2, since we add symmetric links
                    total_degree += 2

                    if links_made == m:
                        log.debug(f"\t******All {m} links made for node {i}!!")
                        break

            log.debug(f"We've iterated through all the nodes. links made: {links_made} vs {m}")

    # Data will be all ones
    nnz = len(row)
    data = np.ones(nnz)

    # Generate the sparse matrix only at the end, from the coordinate lists
    graph = coo_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)

    t2 = time.time()
    print(f"Time to generate AB model(n={n}, m={m}) with version 2: {t2-t1} s")

    return graph.tocsr()


def main():
    t1 = time.time()

    log.setLevel(logging.INFO)
    # log.setLevel(logging.DEBUG)

    n = 1039
    m = 5

    graph = generate_BA_graph_ensure_m_edges(n, m)

    t2 = time.time()
    print(f"Time to generate AB model(n={n}, m={m}) with version 2: {t2-t1} s")

    print(f"Shape of graph: {graph.shape}")
    print(f"Graph nnz: {graph.nnz}")

    print(graph.A)

    A_minus_AT = graph - graph.T

    print(f"\nIs A symmetric? (Is A - A.T all zeros?): {A_minus_AT.nnz == 0}")

    degrees = [sum(line) for line in graph]
    print(f"degrees: {degrees}")


if __name__ == "__main__":
    main()
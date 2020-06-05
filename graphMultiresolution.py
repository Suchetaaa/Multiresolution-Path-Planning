"""
Create and display graph pyramids
for use in multiresolution analysis

Author: Shashwat Shukla
Date: 4th June 2020
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting, utils, reduction
from scipy import sparse, stats
from scipy.sparse import linalg
from mpl_toolkits.mplot3d import Axes3D

# Fix random seeed for reproducability
np.random.seed(0)


# Sparsify the graph
def sparsifyGraph(M, epsilon, maxiter=20):

    # Test the input parameters
    if isinstance(M, graphs.Graph):
        L = M.L
    else:
        L = M

    N = np.shape(L)[0]

    if not 1./np.sqrt(N) <= epsilon < 1:
        raise ValueError('sparsifyGraph: Epsilon out of required range')

    # Not sparse
    resistance_distances = utils.resistance_distance(L).toarray()

    # Get the Weight matrix
    if isinstance(M, graphs.Graph):
        W = M.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0

    W = sparse.coo_matrix(W)
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))

    # Calculate the new weights
    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)

    for i in range(maxiter):
        C0 = 1 / 30.
        C = 4 * C0
        q = round(N * np.log(N) * 9 * C**2 / (epsilon**2))

        results = stats.rv_discrete(values=(np.arange(np.shape(Pe)[0]), Pe)).rvs(size=int(q))
        spin_counts = stats.itemfreq(results).astype(int)
        per_spin_weights = weights / (q * Pe)

        counts = np.zeros(np.shape(weights)[0])
        counts[spin_counts[:, 0]] = spin_counts[:, 1]
        new_weights = counts * per_spin_weights

        sparserW = sparse.csc_matrix((new_weights, (start_nodes, end_nodes)),
                                     shape=(N, N))
        sparserW = sparserW + sparserW.T
        sparserL = sparse.diags(sparserW.diagonal(), 0) - sparserW

        if graphs.Graph(sparserW).is_connected():
            break
        elif i == maxiter - 1:
            logger.warning('Despite attempts to reduce epsilon, sparsified graph is disconnected')
        else:
            epsilon -= (epsilon - 1/np.sqrt(N)) / 2.

    if isinstance(M, graphs.Graph):
        sparserW = sparse.diags(sparserL.diagonal(), 0) - sparserL
        sparserW = (sparserW + sparserW.T) / 2.
        Mnew = graphs.Graph(sparserW, coords = M.coords)
    else:
        Mnew = sparse.lil_matrix(sparserL)

    return Mnew



# Compute the Kron Reduction
def kronReduction(G, ind):

    if isinstance(G, graphs.Graph):
        L = G.L
    else:
        L = G

    N = np.shape(L)[0]
    ind_comp = np.setdiff1d(np.arange(N, dtype=int), ind)

    L_red = L[np.ix_(ind, ind)]
    L_in_out = L[np.ix_(ind, ind_comp)]
    L_out_in = L[np.ix_(ind_comp, ind)].tocsc()
    L_comp = L[np.ix_(ind_comp, ind_comp)].tocsc()

    Lnew = L_red - L_in_out.dot(linalg.spsolve(L_comp, L_out_in))

    # Make the laplacian symmetric if it is almost symmetric
    if np.abs(Lnew - Lnew.T).sum() < np.spacing(1) * np.abs(Lnew).sum():
        Lnew = (Lnew + Lnew.T) / 2.

    if isinstance(G, graphs.Graph):
        # Suppress the diagonal
        Wnew = sparse.diags(Lnew.diagonal(), 0) - Lnew
        Snew = Lnew.diagonal() - np.ravel(Wnew.sum(0))
        if np.linalg.norm(Snew, 2) >= np.spacing(1000):
            Wnew = Wnew + sparse.diags(Snew, 0)

        # Remove diagonal for stability
        Wnew = Wnew - Wnew.diagonal()

        coords = G.coords[ind, :] if len(G.coords.shape) else np.ndarray(None)
        Gnew = graphs.Graph(W=Wnew, coords=coords, lap_type=G.lap_type,
                            plotting=G.plotting, gtype='Kron reduction')
    else:
        Gnew = Lnew

    return Gnew



# Compute a graph pyramid usung sequential Kron reduction and sparsification
def multiresolution(G, levels, sparsify=True):

    sparsify_eps = min(10. / np.sqrt(G.N), 0.3)    
    reg_eps=0.005

    G.estimate_lmax()

    Gs = [G]
    Gs[0].mr = {'idx': np.arange(G.N), 'orig_idx': np.arange(G.N)}

    for i in range(levels):

        if hasattr(Gs[i], '_U'):
            V = Gs[i].U[:, -1]
        else:
            V = linalg.eigs(Gs[i].L, 1)[1][:, 0]

        V *= np.sign(V[0])
        ind = np.nonzero(V >= 0)[0]

        Gs.append(kronReduction(Gs[i], ind))

        if sparsify and Gs[i+1].N > 2:
            Gs[i+1] = sparsifyGraph(Gs[i+1], min(max(sparsify_eps, 2. / np.sqrt(Gs[i+1].N)), 1.))

        Gs[i+1].estimate_lmax()

        Gs[i+1].mr = {'idx': ind, 'orig_idx': Gs[i].mr['orig_idx'][ind], 'level': i}

        L_reg = Gs[i].L + reg_eps * sparse.eye(Gs[i].N)
        Gs[i].mr['K_reg'] = kronReduction(L_reg, ind)
        Gs[i].mr['green_kernel'] = filters.Filter(Gs[i], lambda x: 1./(reg_eps + x))

    return Gs



G = graphs.SwissRoll(N=1000, seed=42)
levels = 5
Gs = multiresolution(G, levels, sparsify=True)

fig = plt.figure(figsize=(10, 2.5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    plotting.plot_graph(Gs[i+1], ax=ax)
    _ = ax.set_title('Pyramid Level: {} \n Number of nodes: {} \n Number of edges: {}'.format(i+1, Gs[i+1].N, Gs[i+1].Ne))
    ax.set_axis_off()
fig.tight_layout()
plt.show()

G = graphs.Sensor(1200, distribute=True)
levels = 5
Gs = multiresolution(G, levels, sparsify=True)

fig = plt.figure(figsize=(10, 2.5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1) # , projection='3d'
    plotting.plot_graph(Gs[i+1], ax=ax)
    _ = ax.set_title('Pyramid Level: {} \n Number of nodes: {} \n Number of edges: {}'.format(i+1, Gs[i+1].N, Gs[i+1].Ne))
    ax.set_axis_off()
fig.tight_layout()
plt.show()
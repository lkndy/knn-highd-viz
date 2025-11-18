Below is a **research-grade, very detailed design and implementation document** for a clean-room C++ implementation of **“Visualizing large-scale high-dimensional data via hierarchical embedding of KNN graphs” (Zhu et al., 2021)**. It stays strictly faithful to the algorithmic content of the paper: same objective, same hierarchy construction idea, same gradient sharing assumptions, same complexity targets, and same parameter roles.

---

# 1. Scope and Objectives

## 1.1 Goal

Implement in C++ a library and CLI tool that:

1. Takes as input an (N \times D) high-dimensional dataset (X = {x_i}_{i=1}^N).
2. Constructs an approximate **K-NN graph** (typically (K = 100)).
3. Computes an embedding (Y = {y_i}_{i=1}^N) in 2D (or optionally 3D) that:

   * Minimizes the Kullback–Leibler divergence between probability distributions (P) in high-dimensional space and (Q) in low-dimensional space, but optimized via the **LargeVis-like edge-sampling objective with negative sampling**.
   * Uses a **multi-level graph hierarchy** (coarsening + hierarchical refinement) to provide good initializations and avoid poor local minima.
   * Uses **cluster-based gradient approximation** (group-based gradient sharing) to reduce computation while preserving quality.

The implementation must respect the **computational complexity** described in the paper:

* Graph coarsening + hierarchy: (O(KN)).
* Optimization: (O(TNM)), where:

  * (T) is proportional to the number of passes (or epochs).
  * (M) is the number of negative samples per positive (typically (M = 5)).

Total is (O(KN + TNM)), linear in (N).

## 1.2 Non-goals

* No alternative objectives (e.g. UMAP, t-SNE BH-tree, FIt-SNE).
* No spectral initialization.
* No GPU-specific algorithms (paper mentions GPU in related work, but the method itself is CPU).
* No reweighted coarse graphs; coarse graphs are structural-only, as in the paper.

---

# 2. Mathematical Foundation

## 2.1 High-Dimensional Similarities (P)

Given a dataset (X = {x_i \in \mathbb{R}^D}), we build a K-NN graph, and define conditional probabilities only over neighbors:

[
p_{i|j} =
\begin{cases}
\displaystyle \frac{\exp(-|x_i - x_j|^2 / (2\sigma_i^2))}
{\sum_{k\neq i} \exp(-|x_i - x_k|^2/(2\sigma_i^2))},
& j \in NN_K(i),[0.8em]
0, & \text{otherwise.}
\end{cases}
]

The symmetric joint probability:

[
p_{ij} = \frac{p_{i|j} + p_{j|i}}{2N}.
]

The bandwidth (\sigma_i) is chosen by a binary search so that a prescribed **perplexity** (e.g. 50) is achieved when summing over neighbors.

We store (p_{ij}) **only for edges** in the (symmetrized) K-NN graph.

---

## 2.2 Low-Dimensional Similarities (Q)

The low-dimensional embedding (Y = {y_i \in \mathbb{R}^d}) (with (d = 2) or (3)) uses a heavy-tailed kernel:

[
q_{ij} = \frac{(1 + |y_i - y_j|^2)^{-1}}
{\displaystyle \sum_{a\neq b}(1 + |y_a - y_b|^2)^{-1}}.
]

As in LargeVis, we do not need the fully normalized (q_{ij}) in the optimization; the objective will be reformulated in terms of per-edge attractive terms and sampled negatives.

---

## 2.3 Objective with Edge and Negative Sampling

The KL divergence between (P) and (Q) is:

[
\mathrm{KL}(P|Q)
= \sum_{i}\sum_{j} p_{ij} \log\frac{p_{ij}}{q_{ij}}
= \sum_{i}\sum_{j} p_{ij}\log p_{ij} - \sum_{i}\sum_{j} p_{ij}\log q_{ij}.
]

The first term is constant in (Y); minimizing KL is equivalent to:

[
\max \sum_{i,j} p_{ij} \log q_{ij}.
]

Following LargeVis, Zhu et al. use a **negative sampling reformulation**:

[
\max \sum_{(i,j)\in E} p_{ij}
\left[
\log q_{ij} + \sum_{k=1}^{M} \gamma \log(1 - q_{ijk})
\right],
]

where (q_{ijk}) denotes the similarity between (i) and a sampled negative (j_k), and (\gamma) balances positive vs. negative terms.

**Gradient (w.r.t. (y_i))**, for a sampled positive edge ((i,j)) and a set of negatives ({j_k}_{k=1}^M):

[
\mathrm{d}y_i =
-\frac{2 p_{ij}(y_i - y_j)}{1 + |y_i - y_j|^2}
+
\sum_{k=1}^{M}
\frac{2\gamma p_{ij}(y_i - y_{jk})}
{|y_i - y_{jk}|^2(1 + |y_i - y_{jk}|^2)}.
]

This is exactly Eq. (8) in the paper.

---

## 2.4 Cluster-Based Gradient Approximation

Groups (v) are generated through graph coarsening. For all points (y_i, y_j \in v), the paper makes two assumptions:

* **A1 (Probability similarity)**: (p_{i,k} \approx p_{j,k}) for all neighbors (k).
* **A2 (Embedding closeness)**: (y_i \approx y_j).

Under these assumptions, using shared positive/negative samples for the group members yields:

[
\mathrm{d}y_i \approx \mathrm{d}y_j \quad \forall y_i,y_j \in v.
]

Thus, the group gradient:

[
\mathrm{d}v = \frac{1}{|v|}\sum_{i\in v}\mathrm{d}y_i \approx \mathrm{d}y_r,
]

where (r) is any representative in (v). This reduces the cost of computing the group gradient from O(|v| M) to O(M).

---

# 3. Data Model and File Formats

## 3.1 Input Data

* Binary or text format; we assume a generic interface:

  * `N`: number of points, 32-bit signed integer.
  * `D`: dimension, 32-bit signed integer.
  * Data: `N * D` float32 values, row-major.

Alternative formats (e.g. `.fvecs`, `.npy`) can be supported via adapters, but are not part of the core algorithmic contribution.

## 3.2 Output Embeddings

Embeddings:

* `N` rows of `d` floats (`d = 2` by default).
* Binary + optional CSV:

  * Binary: raw float32, row-major, N × d.
  * CSV: `id, y0, y1[, y2]`.

---

# 4. Core Data Structures (C++)

We aim for minimal, clear structures aligned with the theoretical model.

## 4.1 CSR Graph Representation

```cpp
struct CSRGraph {
    // Number of vertices
    std::size_t num_vertices;

    // CSR arrays
    std::vector<std::uint32_t> indptr;   // size = num_vertices + 1
    std::vector<std::uint32_t> indices;  // size = num_edges
    std::vector<float>         pij;      // size = num_edges, symmetric p_ij

    // Convenience: degree(i) = indptr[i+1] - indptr[i]
};
```

Assumptions:

* Graph is undirected. Each undirected edge (i,j) appears exactly once in the CSR representation, or appears in both directions with consistent weights; the implementation must be consistent throughout (we choose one representation and document it).

## 4.2 Hierarchy Levels

```cpp
struct Level {
    CSRGraph graph;               // G^l
    std::vector<float> Y;         // embedding, size = num_vertices * dim

    // Group / coarsening info at this level:
    // gid[v]: group ID at this level (used for gradient sharing)
    std::vector<std::uint32_t> gid;  

    // Parent mapping: vertex at this level to vertex at next coarser level
    std::vector<std::uint32_t> parent; 

    // Children mapping: vertex u in coarser level -> list of vertices v in finer level
    std::vector<std::vector<std::uint32_t>> children;
};

struct Hierarchy {
    std::vector<Level> levels; // 0 = finest, L = coarsest
};
```

Each level stores enough information to:

* Prolongate embeddings from coarse to fine.
* Identify groups for gradient-sharing within that level.

---

# 5. K-NN Graph Construction

## 5.1 Requirements

* Approximate K-NN graph for large N (up to millions).
* K typically 100 (as in paper’s experiments).
* EFANNA-like approach: multiple randomized trees + NN-Descent refinement.
* Time complexity approx. (O(N K)) per pass; typically a constant number of passes.

## 5.2 Algorithm (High-Level)

We implement a simplified EFANNA-style approach:

1. **Initialization**:

   * Build several random projection trees or KD-trees over X.
   * For each data point i, initialize a candidate neighbor list of size K using:

     * Points in same leaf nodes across trees; and/or
     * Randomly sampled points.

2. **NN-Descent Refinement**:

   * Repeat for P passes:

     * For each node i (in parallel):

       * For each current neighbor j of i:

         * For each neighbor k of j:

           * Add k as candidate neighbor of i if it improves distance.
     * After merging candidates, keep best K neighbors per node.

3. **Symmetrization**:

   * For each directed edge i→j:

     * Ensure at least one undirected edge {i,j} exists.
   * Optionally maintain symmetric adjacency: i has j and j has i.

## 5.3 Pseudocode

```pseudo
ALGORITHM BuildApproxKNN(X, N, D, K):
    // Step 1: initialize neighbor lists with EFANNA-like approach
    init_neighbors = InitializeByRandomTrees(X, N, D, K)

    neighbors = init_neighbors
    for pass = 1 to P:
        neighbors_new = neighbors
        for i in 0..N-1:
            for each j in neighbors[i]:
                for each k in neighbors[j]:
                    if j == k: continue
                    if k not in neighbors_new[i]:
                        dist = ||x_i - x_k||^2
                        if neighbors_new[i] has < K entries OR dist < worst_dist(neighbors_new[i]):
                            insert (k, dist) into neighbors_new[i]
                            if size(neighbors_new[i]) > K:
                                remove worst
        neighbors = neighbors_new

    // Step 2: symmetrize to produce an undirected graph
    G = empty CSRGraph
    for i in 0..N-1:
        for each j in neighbors[i]:
            add undirected edge (i, j) to G

    return G
```

Details such as tree-building and candidate merging can follow standard EFANNA / NN-Descent recipes; those are implementation details, not conceptual deviations.

---

# 6. Probability Computation on the K-NN Graph

## 6.1 Per-node (\sigma_i) via Perplexity

For each node i:

1. Let (N_i) be its K neighbors.
2. We want perplexity ≈ targetP (e.g. 50):

[
\text{Perp}(P_i) = 2^{H(P_i)}, \quad
H(P_i) = -\sum_{j \in N_i} p_{i|j} \log_2 p_{i|j}.
]

3. Use binary search on (\sigma_i \in (0, \infty)) to achieve the target perplexity.

### Pseudocode

```pseudo
ALGORITHM ComputeSigmaForNode(i, neighbors[i], targetP):
    // neighbors[i] contains (j, d_ij^2) distances
    lo = -INF, hi = +INF
    beta = 1.0    // beta = 1 / (2 sigma^2)
    tolerance = 1e-5

    // precompute distances d_ij^2
    d2 = [dist^2 for each neighbor j]

    for iter = 1..max_iter:
        // compute p_ij with current beta
        sum_exp = 0
        for idx in neighbors[i]:
            sum_exp += exp(-beta * d2[idx])
        p = [exp(-beta * d2[idx]) / sum_exp]

        // compute entropy H in log2
        H = 0
        for p_ij in p:
            if p_ij > 0:
                H += -p_ij * log2(p_ij)
        perp = 2^H

        if |perp - targetP| < tolerance:
            break

        if perp > targetP:
            // too high perplexity -> beta too small -> increase beta
            lo = beta
            if hi == +INF:
                beta *= 2
            else:
                beta = (beta + hi) / 2
        else:
            // perp < targetP -> beta too large -> decrease beta
            hi = beta
            if lo == -INF:
                beta /= 2
            else:
                beta = (beta + lo) / 2

    sigma_i = sqrt(1.0 / (2 * beta))
    return sigma_i
```

## 6.2 Computing (p_{i|j}) and (p_{ij})

Once (\sigma_i) is known:

```pseudo
ALGORITHM ComputeProbabilities(G, X, targetP):
    for each i in vertices(G):
        sigma_i = ComputeSigmaForNode(i, neighbors[i], targetP)

        sum_exp = 0
        for each neighbor j of i:
            d2 = ||x_i - x_j||^2
            w_ij = exp(-d2 / (2 * sigma_i^2))
            store temp_ij = w_ij
            sum_exp += w_ij

        for each neighbor j of i:
            p_i_given_j = temp_ij / sum_exp
            store p_i_given_j (in some temporary structure)

    // symmetrize
    for edge (i,j) in undirected edges:
        p_ij = (p_i_given_j + p_j_given_i) / (2N)
        store p_ij in CSRGraph.pij
```

We now have (p_{ij}) for each edge; this is the probability used in the LargeVis-style objective.

---

# 7. Multi-Level Graph Construction (Coarsening)

## 7.1 Requirements

From the paper:

* Use **multi-level representation** to capture global structure.
* Build a sequence of graphs (G^0, G^1, ..., G^L) with decreasing node counts.
* Use **km-NN graph** (with (k_m \le K)) for grouping during coarsening, e.g. (k_m = 3).
* No new weights are computed; coarse graphs re-use structure.

Shrink factor:

[
|V^{l+1}| \le \rho |V^l|,\quad \rho = 0.8.
]

Stop coarsening when ( |V^{l+1}| > \rho |V^l| ) would fail or node count too small.

## 7.2 Grouping Procedure at a Level

Given (G^l = (V^l, E^l)):

1. Mark all vertices in (V^l) as unassigned.
2. Randomly permute vertices.
3. For each vertex (v \in V^l) in random order:

   * If v is unassigned:

     * Create a new group g.
     * Assign v to g.
     * From v’s adjacency list, pick up to km neighbors that are unassigned and assign them to g.
4. Once all vertices have been visited, each vertex belongs to exactly one group.

## 7.3 Coarsened Graph Construction

We create (G^{l+1}) as follows:

* Each group g becomes a vertex u in (V^{l+1}).
* For edges:

  * For any edge ((i,j) \in E^l) where i belongs to group g1 and j to group g2:

    * If (g1 \neq g2), add an edge ((g1, g2)) to (E^{l+1}) (if not already present).

Note: We do **not** recompute weights or aggregate p_{ij}; we only define structure. For optimization, we propagate probabilities from the finest graph (see Section 9).

## 7.4 Pseudocode

```pseudo
ALGORITHM CoarsenLevel(G^l, km, rho):
    N_l = |V^l|
    unassigned = {true for i in 0..N_l-1}
    order = random_permutation(0..N_l-1)

    gid = array of size N_l  // group id per vertex
    num_groups = 0

    for v in order:
        if not unassigned[v]: continue
        g = num_groups++
        gid[v] = g
        unassigned[v] = false

        neighbors = first_km_unassigned_neighbors(G^l, v, km, unassigned)
        for u in neighbors:
            gid[u] = g
            unassigned[u] = false

    // Build coarse graph
    G^{l+1} = empty CSRGraph with |V^{l+1}| = num_groups
    // we will map edges
    for each edge (i, j) in E^l:
        g1 = gid[i]; g2 = gid[j]
        if g1 == g2: continue
        add undirected edge (g1, g2) to G^{l+1} (avoid duplicates)

    // Build parent/children mappings
    parent[i] = gid[i] for all i
    children[g] = { i | gid[i] == g }

    // Check shrink condition
    if num_groups > rho * N_l:
        stop coarsening (this is last level)

    return G^{l+1}, parent, children, gid
```

Repeated across levels, we get a **hierarchy** of graphs.

---

# 8. Hierarchy Data Assembly

Using the coarsening routine:

```pseudo
ALGORITHM BuildHierarchy(G^0, km, rho):
    H.levels.clear()
    Level0.graph = G^0
    H.levels.push_back(Level0)

    l = 0
    while true:
        G^l = H.levels[l].graph
        if |V^l| is small enough: break

        (G^{l+1}, parent, children, gid) = CoarsenLevel(G^l, km, rho)

        Level_{l+1}.graph   = G^{l+1}
        Level_{l+1}.parent  = parent  // parent at level l -> level l+1
        Level_{l+1}.children = children
        Level_{l+1}.gid     = gid     // group at level l for gradient sharing

        H.levels.push_back(Level_{l+1})
        l = l + 1

    return H
```

Note that:

* `gid` at level l defines **groups used in cluster-based gradient at level l**.
* `parent` and `children` connect level l and l+1 for **embedding prolongation**.

---

# 9. Sampling Mechanisms

## 9.1 Edge Sampling (Positive Samples)

We need to sample edges (i,j) proportional to p_{ij}.

Implementation:

* Build a **global alias table** over all edges (each edge index e is associated to p_{ij}).
* Sampling an edge is (O(1)).

Data structure:

```cpp
struct AliasTable {
    std::vector<float> prob;    // [0,1], final probabilities
    std::vector<std::uint32_t> alias;

    void build(const std::vector<double>& weights);
    std::uint32_t sample(std::mt19937_64& rng) const;
};

struct EdgeSampler {
    AliasTable alias;
    const CSRGraph* graph;

    void init(const CSRGraph& G) {
        graph = &G;
        std::vector<double> w(G.indices.size());
        for (std::size_t e = 0; e < G.indices.size(); ++e)
            w[e] = static_cast<double>(G.pij[e]);  // p_ij > 0
        alias.build(w);
    }

    // Returns (i, j, edge_index e)
    InlineEdge sample(std::mt19937_64& rng) const;
};
```

`InlineEdge` encodes (source, destination, edge index).

## 9.2 Negative Sampling

Negative nodes are sampled proportional to **degree** (d_j).

* For each vertex j, weight (w_j = d_j = \text{degree}(j)).
* Build alias over vertices.

```cpp
struct NegSampler {
    AliasTable alias;
    void init(const CSRGraph& G) {
        std::vector<double> w(G.num_vertices);
        for (std::size_t i=0; i<G.num_vertices; ++i)
            w[i] = static_cast<double>(G.indptr[i+1] - G.indptr[i]);
        alias.build(w);
    }

    std::uint32_t sample(std::mt19937_64& rng) const;
};
```

---

# 10. Optimization: LargeVis Objective on a Level

## 10.1 Per-Edge Gradient Calculations

Compute:

* (d^2 = |y_i - y_j|^2).
* For a positive:

[
g_{\text{pos}}(i) = -\frac{2 p_{ij} (y_i - y_j)}{1 + d^2}.
]

* For a negative sample k:

[
g_{\text{neg},k}(i) =
\frac{2\gamma p_{ij}(y_i - y_{k})}
{|y_i - y_{k}|^2 (1 + |y_i - y_{k}|^2)}.
]

Total gradient for i:

[
g(i) = g_{\text{pos}}(i) + \sum_{k=1}^M g_{\text{neg},k}(i).
]

In practice, we also update y_j and y_k symmetrically (depending on chosen variant); the paper focuses on gradient w.r.t. y_i; empirically, we follow LargeVis and update both endpoints in a balanced way.

## 10.2 Numerically Stable Implementation

In C++:

* Enforce (d^2 \ge \epsilon) for small epsilon (e.g. 1e-12).
* Clip (|\Delta y|) if needed.

Pseudo:

```pseudo
FUNCTION GradPos(y_i, y_j, p_ij):
    d2 = ||y_i - y_j||^2
    d2 = max(d2, eps)
    inv = 1.0 / (1.0 + d2)
    return -2.0 * p_ij * (y_i - y_j) * inv

FUNCTION GradNeg(y_i, y_k, p_ij, gamma):
    d2 = ||y_i - y_k||^2
    d2 = max(d2, eps)
    denom = d2 * (1.0 + d2)
    denom = max(denom, eps2)
    coeff = 2.0 * gamma * p_ij / denom
    return coeff * (y_i - y_k)
```

---

# 11. Cluster-Based Gradient Sharing at a Level

## 11.1 Group Definition

At level l, we have:

* `gid[i]` for each vertex i in (V^l).
* Groups: sets (v_g = { i \mid gid[i] = g}).

We treat each group (v_g) as a **unit** for sampling:

* One positive edge and M negative samples per group.
* Compute gradient for a single representative r ∈ v_g.
* Apply the same gradient to all members in v_g.

## 11.2 Pseudocode (Group Update)

```pseudo
ALGORITHM GroupStep(Level^l, edgeSampler, negSampler, gamma, M, lr):
    for each group g in 0..G_l-1:
        // select representative
        r = pickRepresentative(v_g) // e.g. smallest index in group

        // sample positive edge incident to r
        (i, j, e) = sampleEdgeIncidentTo(r, edgeSampler, Level^l.graph)
        // ensure that i == r (or swap)
        if i != r: swap(i, j)

        // sample negatives
        neg_nodes = []
        for k = 1..M:
            n = negSampler.sample()
            neg_nodes.append(n)

        // compute gradient for r
        gy = zero-vector(dim)
        gy += GradPos(y[i], y[j], p_ij(e))
        for n in neg_nodes:
            gy += GradNeg(y[i], y[n], p_ij(e), gamma)

        // apply gradient to all nodes in group
        for u in v_g:
            y[u] = y[u] - lr * gy
```

Note:

* We may also update j and each n, but for strict adherence to Eq. (8) we focus on updating y_i; a symmetric variant can update y_j and y_n similarly with appropriate scaled gradients.
* The representative selection does not change the theoretical assumption (A1, A2).

---

# 12. Hierarchical Refinement Procedure

## 12.1 Coarsest-Level Initialization

At coarsest level (L):

```pseudo
for i in 0..|V^L|-1:
    for d in 0..dim-1:
        Y^L[i,d] ~ Normal(0, sigma_init^2)  // e.g. small sigma_init
```

## 12.2 Prolongation (Coarse→Fine)

For level l from L down to 1:

```pseudo
ALGORITHM Prolongate(Level_{l}, Level_{l-1}):
    // parent mapping: parent_{l-1}[i] gives vertex in level l
    for i in 0..|V^{l-1}|-1:
        p = Level_{l-1}.parent[i]
        for d in 0..dim-1:
            Y^{l-1}[i,d] = Y^{l}[p,d]
```

## 12.3 Optimization Schedule per Level

Let:

* `T_total ≈ 500 * N` iterations total (paper choice).
* We distribute iterations across levels proportionally to node counts.

Example:

* Level l: `T_l = ceil(500 * |V^l|)` steps.

Parameter γ:

* Coarser levels: γ = 7.
* Finest level (0): γ = 1 (for “early exaggeration” style effect as described).

Learning rate:

* Simple schedule: constant lr per level or decayed linearly or stepwise.

## 12.4 Full Hierarchical Optimization Pseudocode

```pseudo
ALGORITHM HierarchicalOptimize(Hierarchy H, M, base_lr):
    L = H.levels.size() - 1

    // Coarsest level init
    RandomInit(H.levels[L].Y)

    for l = L down to 0:
        if l < L:
            Prolongate(H.levels[l+1], H.levels[l])

        // initialize samplers at level l
        edgeSampler.init(H.levels[l].graph)   // use p_ij as weights
        negSampler.init(H.levels[l].graph)    // degree-based

        // pick gamma depending on level
        if l == 0:
            gamma = 1.0
        else:
            gamma = 7.0

        T_l = 500 * |V^l|  // or some proportion

        for t = 1..T_l:
            // one pass over all groups (order can be randomized)
            GroupStep(H.levels[l], edgeSampler, negSampler,
                      gamma, M, lr_t)
            // update learning rate if needed
            lr_t = UpdateLearningRate(base_lr, t, T_l)
```

At the end, `H.levels[0].Y` is the final embedding.

---

# 13. Complexity Analysis

Let:

* `N = |V^0|`, `K` = K-NN degree, `M` = # negatives, `T` = iterations per node.

### 13.1 Graph Coarsening

Coarsening cost per level l:

* Visit all nodes and edges once → (O(|V^l| + |E^l|)).
* With (|V^{l+1}| \le \rho|V^{l}|) and similar shrink in edges, total coarsening cost:

[
\sum_{l=0}^{L-1} O(|V^l| + |E^l|) \le O(|V^0| + |E^0|) \sum_{l=0}^{\infty} \rho^l
= O((K+1)N).
]

Since (|E^0|\approx K N).

### 13.2 Optimization

Per-step cost at level l:

* For each group: we compute:

  * 1 positive pair gradient.
  * M negative pair gradients.
* Total ~ (|V^l|) groups (since average group size is O(1)).

So per iteration cost is (O(|V^l| M)). With `T_l ∝ |V^l|`, total per level:

[
O(T_l |V^l| M) = O(c |V^l|^2 M)
]

But the paper chooses `T_l = 500|V^l|`; thus the total across levels:

[
O\left(M \sum_l |V^l|^2 \right).
]

However, they consider **overall iteration count referenced to original N**, and with group sharing and coarse graphs the **practical** cost dominated by finest levels, approximated as:

[
O(T N M).
]

Given their empirical choice (T=500) is independent of N, complexity is effectively linear in N for fixed K, M, and T.

In any case, implementation must be structured to avoid additional factors beyond those in the paper.

---

# 14. Parallelization Strategy (Optional, Non-Algorithmic)

Parallelization is optional and must preserve semantics:

* K-NN search: parallelize per node.
* Coarsening: parallelizing group assignment is tricky; easier is to parallelize edge scanning for coarse-edge construction.
* Optimization:

  * Partition groups across threads.
  * Use Hogwild-style updates to Y (2D embedding) or per-thread buffers with periodic reduction.

Constraints:

* Avoid race conditions on Y updates that cause instability; small step sizes and randomization reduce conflicts.
* The algorithm remains conceptually the same.

---

# 15. API and Module Boundaries

## 15.1 Core C++ Namespaces

* `hknn::io`   – loading/saving datasets and embeddings.
* `hknn::graph` – KNN construction, CSR structures, coarsening.
* `hknn::embed` – hierarchy building, samplers, optimization.

## 15.2 Main Functions

```cpp
// High-level embedding call
Embedding run_hierarchy_embedding(
    const float* X, std::size_t N, std::size_t D,
    std::size_t K = 100,
    std::size_t target_dim = 2,
    float perplexity = 50.0f,
    std::size_t max_levels = 0, // auto
    int M = 5,
    float gamma_coarse = 7.0f,
    float gamma_fine = 1.0f,
    float base_lr = 200.0f,
    std::uint64_t seed = 12345
);
```

Internally, this function:

1. Builds approximate K-NN graph.
2. Computes p_{ij}.
3. Builds hierarchy.
4. Runs hierarchical optimization.
5. Returns embedding of size N × target_dim.

---

# 16. Testing and Validation

## 16.1 Unit Tests

1. **K-NN construction**:

   * On small N, D, compare approximate neighbors to exact neighbors.
2. **Probability computation**:

   * Verify that (\sum_j p_{i|j} ≈ 1) for each i.
   * Verify that (\sum_{i,j} p_{ij} ≈ 1).
3. **Graph coarsening**:

   * Check that every vertex is in exactly one group.
   * Check that no self-loops in coarse graph (or handle them consistently).
4. **Sampling correctness**:

   * Validate alias tables via frequency histograms.
5. **Gradient correctness**:

   * On a very small graph, compare analytic gradient to finite-difference gradient.

## 16.2 Integration Tests

* Run on MNIST, FMNIST, etc., and check:

  * 10-NN classification accuracy in 2D:

    * Values should be comparable to those reported (within reasonable tolerance).
  * Timing: should show speedup vs. naive per-point gradient (no sharing).

---

# 17. Reproducibility

* Expose RNG seed.
* Log:

  * N, D, K, perplexity.
  * km, ρ, number of levels.
  * M, gamma values, learning rate schedule, T_l per level.
* Write intermediate embeddings to disk if requested.

---

This completes a **research-lab grade, very detailed design** that is aligned with Zhu et al. (2021) and suitable for direct implementation in C++. If you want, I can now:

* Start generating the **actual C++ code** module by module following this spec, or
* Produce more **detailed pseudocode** for specific parts (e.g. the NN-Descent KNN builder, the alias table construction, or the parallel optimization loop).


Below is a complete, paper-faithful implementation design for “Visualizing large-scale high-dimensional data via hierarchical embedding of KNN graphs” (Zhu et al., 2021), written for a clean-room C++ build. It specifies algorithms, data structures, concurrency, numerics, and a Boost-based toolchain. Where essential, I cite the exact statements and equations from the paper.

1. Problem statement and target

Given high-dimensional vectors X=\{x_i\}_{i=1}^N, construct a K-NN graph and compute a 2-D/3-D embedding Y=\{y_i\} by optimizing a LargeVis-style objective with edge sampling and negative sampling. Two accelerants are required:
	1.	A multi-level representation (graph pyramid) to initialize and refine from coarse→fine, built in linear time and without recomputing edge weights for new graphs, which improves initialization and aesthetics.  ￼
	2.	Cluster-based gradient computation (“gradient sharing”): treat a local group as a unit, share positive/negative samples, and forward the representative’s gradient to group members to reduce per-step work from O(nM) to O(M).  ￼  ￼

The algorithm achieves substantial speedups over LargeVis while preserving quality according to the paper’s evaluation.  ￼

2. Mathematical core (implemented objective and gradients)

2.1 Probabilities on the K-NN graph

High-D similarities (restricted to K-NN edges) follow SNE/t-SNE style:
p_{i\mid j}=\frac{\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)}\quad\text{if }j\in \mathrm{NN}K(x_i),\; \text{else }0,
p{ij}=\frac{p_{i\mid j}+p_{j\mid i}}{2N}.
We store p_{ij} only on edges.  ￼  ￼  ￼

Low-D similarities use a heavy-tailed kernel:
q_{ij} \propto (1+\|y_i-y_j\|^2)^{-1}.
(We compute the normalized form when needed in evaluation; optimization uses the LargeVis reformulation below.)  ￼

2.2 LargeVis reformulation with negative sampling

With edge sampling and M negative samples, optimize:
\max \sum_{i,j} p_{ij}\Big[\log q_{ij} + \sum_{k=1}^{M}\gamma\log(1-q_{ijk})\Big],
with gradient (w.r.t. y_i):
\mathrm{d}y_i =
-\frac{2p_{ij}(y_i-y_j)}{1+\|y_i-y_j\|^2}
+\sum_{k=1}^M \frac{2\gamma p_{ij}(y_i-y_{jk})}{\|y_i-y_{jk}\|^2(1+\|y_i-y_{jk}\|^2)}.
This is exactly Eqs. (7)–(8) in the paper. Sampling: positives by p_{ij} (edge sampling), negatives by degree-proportional distribution.  ￼  ￼

2.3 Cluster-based gradient sharing

Let v be a group (cluster) of low-D points. Assume (A1) members share similar neighbor probabilities and (A2) are close in Y; then the sampled positives/negatives and resulting gradients are (approximately) common across the group. The paper justifies forwarding a representative gradient to all members, reducing the work from O(nM) to O(M) for the group. Use Fig. 2’s sharing scheme.  ￼  ￼  ￼

3. Software architecture (C++, Boost-centric)

3.1 Build and dependencies
	•	Toolchain: C++20, CMake ≥ 3.24.
	•	Boost (header-only unless noted): program_options, filesystem (or C++17 fs), container (PMR allocators), random, dynamic_bitset, lockfree, thread, asio (thread_pool), timer, accumulators.
	•	Optional: OpenMP (or stick to Boost.Thread), spdlog for logging (or Boost.Log).

project(hknn_embed CXX)
find_package(Boost REQUIRED COMPONENTS program_options filesystem random thread)
add_library(hknn STATIC ...)
target_compile_features(hknn PRIVATE cxx_std_20)
target_link_libraries(hknn PUBLIC Boost::program_options Boost::filesystem Boost::random Boost::thread)
add_executable(embed_hknn main.cpp)
target_link_libraries(embed_hknn PRIVATE hknn)

3.2 Modules and responsibilities
	•	io/
	•	reader.hpp: mmap/stream dense float arrays (.fvecs, .npy optional).
	•	writer.hpp: write .f32 (raw) and .csv embeddings.
	•	graph/
	•	csr_graph.hpp: CSR adjacency with p_ij weights.
	•	knn_builder.hpp/cpp: exact KD/Ball-tree for small D; NN-Descent (approximate) for large D (in-house).
	•	coarsen.hpp/cpp: build hierarchy (parent/children, groups).
	•	embed/
	•	sampler.hpp/cpp: alias tables for p_{ij}; negative reservoir by degree.
	•	optimizer.hpp/cpp: SGD loop; learning rate schedule; momentum (optional).
	•	sharing.hpp/cpp: group representative selection, shared pos/neg, gradient fan-out.
	•	hierarchy.hpp/cpp: coarse→fine orchestration.
	•	math/
	•	kernels.hpp: stable distance, inverse 1/(1+d^2); small-dim vector ops.
	•	rng.hpp: PCG or xoshiro via Boost.Random wrappers.
	•	cli/
	•	args.hpp: Boost.Program_options definitions.

4. Data structures

// CSR over an undirected K-NN graph (store each undirected edge once)
struct CSR {
  std::vector<uint32_t> indptr;     // |V|+1
  std::vector<uint32_t> indices;    // |E|
  std::vector<float>    pij;        // |E| (symmetrized weights)
};

struct Level {
  CSR g;                             // graph at level l
  // Grouping derived from coarsening; gid[v] == group id at this level.
  std::vector<uint32_t> gid;         // |V|
  // Parent/fine<->coarse mapping
  std::vector<uint32_t> parent;      // |V_l| -> vertex in V_{l+1}
  std::vector<std::vector<uint32_t>> children; // inverse (in next level)
  // Embedding coordinates (SoA by vertex-major)
  std::vector<float> Y;              // |V_l| * dim (dim ∈ {2,3})
};

struct Hierarchy { std::vector<Level> L; /* [0]=finest … [K]=coarsest */ };

Memory considerations: Use pooled/PMR allocators (boost::container::pmr::vector) for large arrays to reduce fragmentation. Favor SoA (contiguous Y) to enable vectorized distance/gradient accumulation.

5. Algorithms and implementation detail

5.1 K-NN construction (from scratch)
	•	Exact (D ≤ ~20, N ≤ ~2e6): a blocked multi-KD-tree with priority-queue queries; parallelize queries via a work-stealing queue (boost::lockfree::queue<uint32_t>).
	•	Approximate (default): NN-Descent:
	•	Start with random neighbor lists (degree K).
	•	Iteratively refine: for each node, consider neighbors-of-neighbors; keep K best; stop when improvement < ε.
	•	Parallelize per-node refinement; use dynamic_bitset to avoid duplicate candidate checks.

Result is a directed K-NN; we symmetrize (keep union) and compute p_{ij} only on kept edges using Eq. (4)–(5) with per-row binary search for \sigma_i to match a target perplexity (restricted to neighbors).  ￼  ￼

5.2 Multi-level representation (graph pyramid)

Goal: construct decreasing-size graphs in linear time without recomputing edge weights for new graphs; use them to initialize and refine layouts coarse→fine.  ￼

Coarsen() (one pass):
	1.	Shuffle vertices; maintain unassigned bitset.
	2.	For each seed v not yet assigned, form a group by pulling its first k_ml still-free neighbors from its adjacency (purely structural).
	3.	Contract groups:
	•	parent[v] = g, push v into children[g].
	•	Add coarse edge (g_u, g_v) if any edge across their member sets exists.
	4.	Stop when the next level would shrink less than \rho (e.g., 0.8), or when |V| is small.

Notes: We do not recompute weights on coarser graphs; the paper’s speed/quality hinge on sharing at optimization time rather than reweighting.  ￼

5.3 Samplers
	•	Positive edge sampler: global alias table over edges with probabilities proportional to p_{ij} (or per-node alias chosen proportional to p_{ij} and then expand to j). Edge sampling “avoids excessive gradient” and is standard in LargeVis.  ￼
	•	Negative sampler: degree-proportional distribution; build a reservoir/alias over vertex IDs with weights d_j.  ￼
Use boost::random::mt19937_64 or PCG; expose seed for reproducibility.

5.4 Low-D kernel and gradients (stable numerics)

inline float inv_one_plus_sqdist(const float* __restrict a,
                                 const float* __restrict b, int dim) {
  float d2 = 0.f;
  for (int t=0;t<dim;++t) { float d = a[t]-b[t]; d2 += d*d; }
  d2 = std::max(d2, 1e-12f);
  return 1.0f / (1.0f + d2);
}

inline void grad_pair_pos(float* __restrict gi,
                          const float* __restrict yi,
                          const float* __restrict yj,
                          float pij, int dim) {
  float inv = inv_one_plus_sqdist(yi,yj,dim);
  for (int t=0;t<dim;++t) gi[t] += -2.0f * pij * (yi[t]-yj[t]) * inv;
}

inline void grad_pair_neg(float* __restrict gi,
                          const float* __restrict yi,
                          const float* __restrict yk,
                          float pij, float gamma, int dim) {
  // matches the denominator in Eq. (8) form
  float d2 = 0.f; for (int t=0;t<dim;++t){ float d=yi[t]-yk[t]; d2+=d*d; }
  d2 = std::max(d2, 1e-12f);
  float inv = 1.0f / (1.0f + d2);
  float denom = d2 * (1.0f + d2);
  float coeff = 2.0f * gamma * pij / std::max(denom, 1e-18f);
  for (int t=0;t<dim;++t) gi[t] += coeff * (yi[t]-yk[t]);
}

This implements the gradient terms in Eq. (8).  ￼

5.5 Cluster-based gradient sharing (per group update)

For each group v at the current level:
	1.	Pick a representative r\in v (e.g., smallest index, or the one with max degree).
	2.	Sample one positive (r,j) by edge sampling and M negatives \{n_k\} by degree sampling.
	3.	Compute \Delta y_r using Eq. (8).
	4.	Fan-out: apply \Delta y_r to all members u\in v.

The paper shows assumptions A1: p_{i,k}\approx p_{j,k} and A2: y_i \approx y_j justify shared samples and approximate equality of gradients across the group (Eq. (11)), enabling the O(M) per-group cost.  ￼  ￼  ￼

Engineering safeguards:
	•	If intra-group spread becomes large (RMS \|y_i-y_r\| > τ), temporarily disable sharing for that group or re-split.
	•	Optionally scale fan-out by a per-member factor \alpha_u\in[0.8,1.2] estimated from local degree mismatch (heuristic).

5.6 Hierarchical refinement loop

Coarsest→finest:
	1.	Coarsest init: Gaussian N(0,\sigma^2) with small \sigma.
	2.	For level \ell (coarse) to \ell-1 (finer):
	•	Initialize Y^{\ell-1} by prolongation: each child inherits its parent’s coordinate.
	•	Run T_\ell epochs of SGD with edge+negative sampling; enable gradient sharing on this level’s grouping.
	•	On the finest level, you may disable sharing in the last epochs for exactness.

The paper emphasizes that multi-level initialization avoids poor local minima and improves aesthetics.  ￼

6. Optimization engine and schedules
	•	SGD orchestration: sharded mini-batches of sampled pairs. Each worker thread processes a shard; updates use:
	•	Hogwild for small dim∈{2,3} (atomic float not required in practice if sparsity is high and batch size small).
	•	Or per-thread buffers (accumulate \Delta Y) and #pragma omp parallel for/thread-pool barrier to apply.
	•	Learning rate: \eta_t = \eta_0/(1+t/T)^\beta with \beta\in[0.5,1]. Allow cosine schedule alternative.
	•	“Early exaggeration” analogue (optional): on coarse levels, increase \gamma and/or iteration budget; on finest, reduce \gamma for polishing (the paper uses \gamma to balance pos/neg).  ￼
	•	Clipping: cap \|\Delta y\| and minimum distance d^2 \ge 10^{-12}.

7. Parallelization (Boost-first)
	•	Thread pool: boost::asio::thread_pool sized to min(N_hw, max(2, N_hw-1)).
	•	Work queues: boost::lockfree::queue<uint32_t> for vertex jobs (K-NN refinement) and group jobs (SGD updates).
	•	RNG: independent boost::random::mt19937_64 engines per thread + counter-based jump for reproducibility.
	•	Timing: boost::timer::cpu_timer segments for profiling (K-NN, coarsen, each level’s SGD).

8. CLI and configuration

embed_hknn \
  --input data.fvecs --N 3000000 --D 100 \
  --K 100 --perplexity 50 \
  --levels_auto 1 --kml 3 --rho 0.8 \
  --epochs_coarse 4 --epochs_fine 8 \
  --mneg 5 --gamma 5.0 \
  --lr 200.0 --schedule poly:beta=0.5 \
  --dim 2 --seed 123 --out reviews_2d.f32

Key flags:
	•	--K, --perplexity → construction of p_{ij} on the K-NN graph (Eq. (4)–(5)).  ￼  ￼
	•	--kml, --rho → grouping granularity & shrink ratio per level.
	•	--mneg, --gamma → negatives M and pos/neg balance (Eq. (7)–(8)).  ￼
	•	--epochs_* per level; --levels_auto builds pyramid until |V| small.

9. File formats
	•	Input: .fvecs (FAISS style) or raw float32 row-major; optional .npy.
	•	Output: .f32 (float32 row-major, N×dim) + .csv (id,x,y[,z]) for quick inspection.

10. Validation and evaluation
	•	Convergence monitors:
	•	Running average of objective proxy (sampled \log q + \gamma\log(1-q)).
	•	Neighbor preservation @k (compute 10-NN in Y vs. X’s labels where available), matching paper’s qualitative claims that 10-NN accuracy is comparable to BH-SNE/LargeVis.  ￼
	•	Unit tests:
	•	CSR integrity (degree sums), symmetry after union.
	•	Probability normalization per row for p_{i\mid j}; symmetry p_{ij}.
	•	Sampler correctness (chi-square over draws).
	•	Gradient check on tiny graphs against finite differences (2D).
	•	Ablations:
	•	Sharing vs. no-sharing runtime and quality deltas (verify expected O(M) per-group effect).  ￼
	•	Coarse→fine vs. random init aesthetics (qualitative like Fig. 3 in paper).  ￼

11. Performance and complexity
	•	K-NN build: O(NK) per pass (NN-Descent), a few passes typical.
	•	Coarsening: single linear sweep per level (edge scan + grouping).
	•	Optimization: Per iteration cost proportional to sampled edges plus M negatives; with sharing, per-group work reduces to O(M).  ￼
	•	Empirical speed: the paper reports ~5× speedup vs. LargeVis for K-NN graph visualization (we target comparable behavior).  ￼

12. Failure modes and remedies
	•	Collapsed clusters / poor separation: increase epochs on coarse levels; raise \gamma on coarse, lower on fine; reduce kml to form tighter groups. (Trade-off between speed and faithfulness.)
	•	Sharing hurts fidelity: detect group spread and adaptively disable sharing for that group; or run final “no-sharing” polish epochs on the finest level.
	•	Stalls (plateaus): cosine LR restart; raise M modestly.

13. Code skeletons (selected)

13.1 Sampler (alias table)

struct Alias {
  std::vector<float> prob;
  std::vector<uint32_t> alias;
  void build(const std::vector<double>& w);
  inline uint32_t draw(BoostRng& rng) const noexcept;
};

struct EdgeSampler {
  Alias global_edges;                 // over [0..|E|-1]
  const CSR* csr = nullptr;
  void init(const CSR& g, const std::vector<float>& pij);
  inline std::pair<uint32_t,uint32_t> sample_edge(BoostRng& rng) const;
};

struct NegSampler {
  Alias nodes;
  void init(const CSR& g) {
    std::vector<double> w(g.indptr.size()-1);
    for (size_t u=0; u<w.size(); ++u) w[u] = double(g.indptr[u+1]-g.indptr[u]); // degree
    nodes.build(w);
  }
  inline uint32_t sample(BoostRng& rng) const { return nodes.draw(rng); }
};

(Edge sampling for positives and degree-proportional negatives per the paper.)  ￼

13.2 One SGD step with sharing

void sgd_groups(Level& L, const CSR& G, const OptimCfg& cfg, Rng& rng) {
  // iterate groups (could be randomized)
  for (uint32_t g = 0; g < num_groups(L); ++g) {
    const uint32_t rep = pick_representative(L, g);
    const auto [i,j] = pos_edge_sample_for(rep, G, rng);  // by p_ij
    std::array<uint32_t,MAX_M> neg;
    for (int k=0;k<cfg.M;++k) neg[k] = neg_sampler.sample(rng);

    float gi[3] = {0,0,0};
    grad_pair_pos(gi, &L.Y[i*dim], &L.Y[j*dim], pij(i,j), dim);
    for (int k=0;k<cfg.M;++k)
      grad_pair_neg(gi, &L.Y[i*dim], &L.Y[neg[k]*dim], pij(i,j), cfg.gamma, dim);

    for (auto u : members(g))            // fan-out
      for (int t=0;t<dim;++t) L.Y[u*dim+t] -= cfg.lr * gi[t];
  }
}

13.3 Hierarchical refinement

for (int ell = Lmax; ell >= 0; --ell) {
  if (ell == Lmax) random_init(level[ell].Y, dim, seed);
  else prolongate(level[ell+1], level[ell]); // copy parent coords to children

  for (int e=0; e<epochs(ell); ++e) {
    parallel_for_groups(level[ell], [&](auto& batch, Rng& rng){
      sgd_groups(level[ell], level[ell].g, optim_cfg, rng);
    });
    maybe_decay_lr(optim_cfg, e);
  }
}

14. Parameters (practical defaults)

While the paper shows parameter roles rather than prescribing universal constants, reasonable starting points are:
	•	K=100, perplexity≈50 (for stable p_{ij} on K-NN; tune to dataset scale).
	•	Coarsening: kml=3, rho=0.8 to shrink ~20% per level.
	•	Optimization: M=5 negatives, \gamma \in [3,7] (higher on coarse; near 1 on finest) to balance attraction/repulsion per Eq. (7).  ￼
Adjust per validation (neighbor preservation and visual inspection).

15. Logging and reproducibility
	•	Fixed --seed for RNG; log all hyperparameters and level sizes.
	•	Time breakdown per stage with boost::timer::cpu_timer.
	•	Dump intermediate embeddings per level (--dump_levels).

16. Deliverables
	•	libhknn.a (+ headers), embed_hknn CLI.
	•	Scripts for MNIST/CIFAR-like demo datasets.
	•	Benchmarks (runtime vs. LargeVis-style baseline; neighbor preservation @10).
	•	Developer docs cross-referencing the cited equations and algorithmic sections.

⸻

Paper cross-references
	•	Abstract & contribution summary (multi-level + gradient approximation; speedup):  ￼  ￼
	•	Prior methods and LargeVis overview:  ￼
	•	SNE/t-SNE objective, p_{ij}, and low-D kernel q_{ij}:  ￼  ￼  ￼
	•	Negative sampling objective and gradient (Eqs. 7–8), edge/negative sampling:  ￼  ￼
	•	Key idea: multi-level initialization + gradient approximation:  ￼
	•	Multi-level representation (linear time; no extra weight computation):  ￼
	•	Cluster-based gradient sharing assumptions and complexity reduction (Eq. 11):  ￼  ￼  ￼

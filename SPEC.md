# embed-cluster -- Specification

## 1. Overview

`embed-cluster` is an embedding-aware clustering library that groups high-dimensional embedding vectors into semantically coherent topics, automatically determines the optimal number of clusters, scores each clustering's quality with silhouette analysis, and labels every cluster with human-readable topic names. It is designed specifically for the characteristics of embedding vectors produced by modern language models -- high dimensionality (768 to 3072 dimensions), cosine-similarity geometry, and semantic structure -- where generic clustering libraries fail or produce poor results without significant hand-tuning.

The gap this package fills is specific and well-defined. Embedding vectors are now the lingua franca of AI applications: every RAG pipeline embeds documents, every semantic search system embeds queries, every memory system embeds thoughts and observations. But embedding vectors are rarely just stored and retrieved -- they are also mined for structure. What topics recur in a document collection? What themes does a knowledge base cover? Which user queries cluster together into support categories? Which memories belong together in episodic groups? These questions all reduce to clustering over embedding vectors.

Generic clustering libraries exist in the npm ecosystem but none address the embedding use case end-to-end. `ml-kmeans` implements Lloyd's algorithm but requires the caller to specify k, chooses random initialization, uses Euclidean distance by default (suboptimal for embeddings), and returns no quality metrics. `skmeans` has the same limitations. `density-clustering` implements DBSCAN and OPTICS but these algorithms require a density-threshold parameter that is nearly impossible to tune correctly in high-dimensional embedding spaces due to the curse of dimensionality. `clusterfck` implements hierarchical clustering but does not scale beyond a few thousand points and produces no topic labels. None of these libraries provide automatic k selection, silhouette scoring, or any mechanism for attaching human-readable labels to the clusters they produce. Every team that wants to cluster embeddings today must wire together their own k-means implementation, their own elbow-method loop, their own silhouette scorer, and their own keyword extractor. The result is fragile, inconsistent, and slow to build.

`embed-cluster` provides a complete, production-ready clustering pipeline in a single `cluster()` call. It runs k-means with k-means++ initialization and cosine distance (the correct metric for unit-normalized embedding vectors), automatically determines the optimal cluster count using the elbow method and silhouette analysis in combination, scores every clustering decision with a per-point and aggregate silhouette score, and labels each cluster using TF-IDF keyword extraction over the associated texts or a caller-supplied labeling function. The API accepts either raw embedding vectors or structured `EmbedItem` objects that pair text with embeddings, and returns a structured `ClusterResult` with every cluster's centroid, members, quality scores, and label. A CLI allows offline exploration of embedding files. Optional dimensionality reduction (PCA) accelerates clustering for very high-dimensional inputs.

---

## 2. Goals and Non-Goals

### Goals

- Provide a `cluster(embeddings, options?)` function that accepts an array of embedding vectors and returns a `ClusterResult` with per-cluster members, centroids, labels, and quality scores.
- Provide a `cluster(items, options?)` overload that accepts an array of `EmbedItem` objects (each pairing a `text` string with an `embedding` vector) and additionally generates topic labels using TF-IDF keyword extraction over the cluster's texts.
- Implement k-means clustering with k-means++ initialization, configurable distance metrics (cosine distance as default, Euclidean as alternative), convergence detection, and multiple restarts with best-of-N selection.
- Implement automatic cluster count detection via the elbow method: run k-means for k = 2 through `maxK`, compute inertia (within-cluster sum of squares) for each k, and programmatically detect the elbow using the kneedle algorithm (maximum curvature of the inertia curve).
- Implement automatic cluster count detection via silhouette analysis: compute the mean silhouette coefficient for each k, and select the k that maximizes the mean silhouette score.
- Combine elbow and silhouette methods into a consensus decision: when both methods agree, use that k; when they disagree, weight silhouette more heavily (as it is a more direct measure of clustering quality) but expose both suggestions in the result.
- Allow callers to specify an exact k, bypassing auto-detection, when the number of clusters is known in advance.
- Implement per-point silhouette coefficient computation: for each point, compute the average intra-cluster distance (a) and the average nearest-cluster distance (b), and return (b - a) / max(a, b). Report per-cluster and aggregate average silhouette scores.
- Generate topic labels for each cluster when text is available: use TF-IDF weighting across the corpus to extract the top terms that most distinguish each cluster from the others.
- Support a caller-supplied `labeler` function as an alternative to built-in TF-IDF labeling, enabling LLM-generated labels or domain-specific terminology.
- Identify outliers: points with a negative silhouette coefficient are candidates for an "unclustered" or outlier designation.
- Export visualization data: 2D PCA projections for each point, cluster colors, and centroid projections, suitable for rendering with d3, plotly, or Observable.
- Provide optional PCA preprocessing for dimensionality reduction before clustering, with configurable target dimensions.
- Provide a CLI (`embed-cluster`) that reads embedding files (JSON), clusters them, and writes the `ClusterResult` to stdout.
- Provide a `findOptimalK(embeddings, options?)` function for callers who want to determine the optimal k independently of clustering.
- Provide a `silhouetteScore(embeddings, labels)` function for callers who have an existing clustering assignment and want to evaluate it.
- Provide a `createClusterer(config)` factory for creating a pre-configured clusterer instance for repeated use.
- Integrate with `embed-cache` (consuming its cached embedding vectors), `embed-drift` (detecting distribution shifts in cluster membership over time), `memory-dedup` (deduplicating semantically similar cluster members), and `fusion-rank` (providing cluster-aware retrieval diversity).
- Zero mandatory runtime dependencies. All algorithms are self-contained JavaScript/TypeScript. Optional peer dependencies for PCA acceleration (ml-pca) and advanced visualization (umap-js).
- Target Node.js 18 and above. Work in any JavaScript environment that supports typed arrays.

### Non-Goals

- **Not an embedding provider.** This package does not call any embedding API. It clusters embedding vectors that the caller has already obtained. Bring your own embedding function, `embed-cache` instance, or OpenAI client.
- **Not a vector database.** This package does not store embedding vectors persistently, index them for nearest-neighbor search, or support metadata filtering. Use Vectra, Pinecone, Weaviate, or Qdrant for that.
- **Not a full NLP pipeline.** TF-IDF keyword extraction provides useful cluster labels for most embedding use cases. This package does not implement full topic models (LDA, NMF), named entity recognition, or part-of-speech tagging. For deep NLP, the custom `labeler` hook is the integration point.
- **Not a real-time clusterer.** This package makes a complete clustering decision over a fixed set of vectors. Incremental clustering (adding new points to an existing clustering without re-running the full algorithm) is out of scope. For live pipelines, re-run clustering periodically on the full corpus.
- **Not a graph clusterer.** This package implements partition-based clustering (k-means). Graph-based or density-based approaches (DBSCAN, OPTICS, Louvain community detection on a similarity graph) are not implemented. These approaches can be better for non-convex cluster shapes but are harder to tune and scale poorly to high-dimensional spaces.
- **Not a supervised classifier.** Clustering is unsupervised. This package finds latent groupings in unlabeled data. It does not train a classifier, assign predefined category labels, or use labeled examples to define cluster boundaries.
- **Not a dimensionality reduction visualization tool.** The package exports PCA 2D coordinates as a convenience for downstream visualization. It does not provide interactive plots, t-SNE, or UMAP (though a UMAP peer-dependency hook is planned for a future version).
- **Not a streaming processor.** The entire input set must be available before clustering begins. Streaming inputs are not supported.

---

## 3. Target Users and Use Cases

### Document Topic Discovery

A team maintains a knowledge base of 10,000 engineering documents, wikis, and runbooks. They want to understand the topical structure of their corpus before building a RAG system: how many distinct topics exist, what are they, and which documents belong to each? They call `cluster(items)` where each item pairs the document text with its pre-computed embedding vector. The result is a `ClusterResult` with k clusters, each labeled with its top TF-IDF keywords ("kubernetes deployment", "authentication oauth", "database migration") and containing the documents assigned to it. The team uses this to validate that their corpus covers the topics they need and to identify gaps.

### Content Taxonomy Construction

A content platform has 50,000 user-generated articles. They want to build a content taxonomy for navigation and recommendation, but doing so manually for 50,000 items is infeasible. They use `embed-cluster` with `auto: true` and `maxK: 50` to discover the taxonomy automatically. The resulting 23 clusters (as detected by the consensus elbow + silhouette method) become the top-level taxonomy categories, labeled by TF-IDF keywords. Outlier detection identifies articles that do not fit cleanly into any category -- these become candidates for manual review or a catch-all "Other" category.

### Semantic Search Result Grouping

A search engine retrieves 50 documents for a query. Before presenting results to the user, it wants to group results by sub-topic ("cluster within search results") so the user can navigate to the facet they care about. The application calls `cluster(results.map(r => ({ text: r.snippet, embedding: r.embedding })), { k: 5 })` with a fixed k to get exactly 5 groups, each with a topic label derived from the snippets' TF-IDF terms. The result is faceted search navigation without requiring any pre-defined taxonomy.

### Memory Consolidation and Episodic Grouping

An AI agent accumulates observations and memories over time, each stored as a text + embedding pair. Periodically, the agent clusters its memories to identify recurring themes, consolidate related memories into summaries, and prune redundant ones. It calls `cluster(memories)` to discover the natural episodic structure of its accumulated knowledge. Clusters with high intra-cluster similarity (high silhouette score) represent coherent episodes; clusters with low scores represent heterogeneous memories that may benefit from further decomposition. The `memory-dedup` package can then be applied within each cluster to eliminate near-duplicates.

### Anomaly and Outlier Detection

A moderation system embeds all submitted content. Most content clusters into normal categories (product reviews, support requests, shipping questions). Outlier points -- content with a negative silhouette coefficient that does not fit any cluster well -- are flagged for human review as potential spam, abuse, or novel categories that the system has not seen before. The `cluster()` result's per-point silhouette scores enable threshold-based flagging without any labeled training data.

### RAG Pipeline Topic Routing

A multi-tenant RAG system serves queries across several knowledge domains (legal, medical, engineering). Rather than maintaining separate vector databases per domain, it clusters all indexed documents and routes incoming queries to the most relevant cluster centroids for retrieval. This reduces the search space and improves retrieval precision for domain-specific queries. The query's embedding is compared to cluster centroids; only the nearest N clusters are searched in full.

### Embedding Drift Monitoring

A pipeline regularly re-embeds and re-clusters its document corpus. By comparing cluster membership and centroid positions across clustering runs over time, it can detect when the corpus' topic distribution has shifted -- new topics emerging, old topics shrinking, or content migrating between clusters. The `embed-drift` package integrates with `embed-cluster`'s `ClusterResult` to compute cluster-level drift scores.

---

## 4. Core Concepts

### Embedding Vector

An embedding vector is a fixed-length array of floating-point numbers that represents a piece of text in a high-dimensional semantic space. Modern language model embedding APIs produce vectors of 768 dimensions (BERT-family), 1024 dimensions (Cohere embed-english-v3.0), 1536 dimensions (OpenAI text-embedding-3-small), or 3072 dimensions (OpenAI text-embedding-3-large). The key property of embedding spaces is that semantic similarity corresponds to geometric proximity: texts with similar meaning have embedding vectors that are close together according to cosine similarity. Clustering over embedding vectors therefore discovers semantic groupings.

### Cosine Distance

For embedding vectors, cosine distance is the preferred distance metric. Cosine similarity between two vectors u and v is:

```
cosine_similarity(u, v) = dot(u, v) / (||u|| * ||v||)
```

Cosine distance is the complement: `cosine_distance(u, v) = 1 - cosine_similarity(u, v)`. Cosine distance ranges from 0 (identical direction) to 2 (opposite direction), but in practice for embedding vectors it falls in the range [0, 1].

The reason cosine distance is preferred over Euclidean distance for embeddings is the curse of dimensionality. In high-dimensional spaces, Euclidean distances between points tend to converge: the ratio of the farthest point's distance to the nearest point's distance approaches 1 as dimensionality increases. This makes Euclidean-distance-based nearest-neighbor queries and k-means centroids poorly discriminative. Cosine similarity is invariant to vector magnitude and measures only directional alignment, which remains meaningful in high-dimensional embedding spaces. Most embedding APIs implicitly encourage cosine distance by returning unit-normalized vectors (or recommending normalization before comparison).

When all input vectors are L2-normalized (||v|| = 1 for all v), cosine distance and Euclidean distance produce identical ranking (though not identical magnitudes), because for unit vectors, dot(u, v) = 1 - 0.5 * ||u - v||^2. In this case, Euclidean k-means and cosine k-means give the same partition. `embed-cluster` normalizes all vectors before clustering as a preprocessing step when `distanceMetric: 'cosine'` (the default), making the standard centroid recomputation step of Lloyd's algorithm equivalent to spherical k-means.

### Centroid

A cluster centroid is the mean of all embedding vectors assigned to that cluster. In Euclidean k-means, the centroid is the arithmetic mean. For cosine-distance k-means on L2-normalized vectors (spherical k-means), the centroid is the arithmetic mean of the cluster members, re-normalized to unit length after each update. The centroid represents the "center of gravity" of the cluster in embedding space -- it is the point from which the average intra-cluster distance is minimized.

The centroid is also useful as a query vector: to find documents most similar to a cluster's topic, a user can search a vector database with the cluster centroid as the query, retrieving documents beyond those in the cluster that are semantically similar to the cluster's topic.

### Inertia

Inertia is the within-cluster sum of squared distances from each point to its cluster centroid. For a clustering C = {C_1, ..., C_k} with centroids {μ_1, ..., μ_k}:

```
inertia = Σ_{i=1..k} Σ_{x ∈ C_i} distance(x, μ_i)^2
```

Inertia measures how compact the clusters are. Lower inertia means tighter clusters. Inertia always decreases as k increases (adding more clusters can only reduce the average distance to a centroid), which is why it cannot be used alone to select k -- it would always suggest using as many clusters as there are points. The elbow method exploits the fact that inertia decreases quickly at first (as meaningful structure is captured) and then more slowly (as additional clusters capture noise), looking for the k where the rate of decrease drops.

### Silhouette Coefficient

The silhouette coefficient for a single point x is:

```
s(x) = (b(x) - a(x)) / max(a(x), b(x))
```

where:
- `a(x)` = mean distance from x to all other points in its cluster (intra-cluster cohesion)
- `b(x)` = mean distance from x to all points in the nearest neighboring cluster (inter-cluster separation)

The silhouette coefficient ranges from -1 to +1:
- **+1**: x is well within its cluster and far from all other clusters. Ideal.
- **0**: x is on or near the decision boundary between two clusters.
- **-1**: x is closer to a neighboring cluster than to its own cluster. It is likely misassigned.

The mean silhouette score for a clustering is the average `s(x)` across all points. A mean silhouette score above 0.5 indicates a good clustering. A score above 0.7 indicates a very good clustering. A score below 0.25 indicates an overlapping or poorly separated clustering where k-means may not be the right algorithm.

Unlike inertia, the silhouette score penalizes both poor cohesion (points far from their centroid) and poor separation (clusters that are close to each other). It therefore peaks at the k that best balances these two forces, making it the most reliable single metric for automatic k selection.

### Elbow Method

The elbow method plots inertia against k and looks for the "elbow" -- the inflection point where the marginal reduction in inertia from adding one more cluster decreases sharply. To the left of the elbow, increasing k captures meaningful structure. To the right, increasing k fragments already-coherent clusters into sub-clusters, reducing inertia by less per additional cluster.

The visual elbow is easy for humans to spot on a plot but difficult to locate programmatically. `embed-cluster` uses the kneedle algorithm (Satopaa et al., 2011) to detect the elbow automatically. The kneedle algorithm:

1. Normalizes the (k, inertia) curve to [0, 1] x [0, 1] by dividing each axis by its range.
2. Computes the "difference curve": at each k, the difference between the normalized inertia and the straight line from (k_min, inertia_min) to (k_max, inertia_max).
3. The knee is the k that maximizes this difference -- the point furthest from the chord connecting the curve's endpoints.

The kneedle algorithm is robust to scale differences in inertia values and returns a single programmatic k recommendation. It is used as one of two inputs to the consensus k-selection decision.

### Automatic Cluster Count Consensus

Neither the elbow method nor silhouette analysis alone is always correct:
- The elbow method is fast but can be ambiguous when the inertia curve is smooth with no sharp inflection.
- Silhouette analysis is more reliable but computationally more expensive (requires pairwise distance computation for every k candidate).

`embed-cluster` computes both and applies a consensus rule:
- If both methods agree (recommend the same k), use that k. Confidence: high.
- If they disagree by 1 (adjacent k values), prefer the silhouette recommendation. Confidence: medium.
- If they disagree by 2 or more, run a tiebreaker: compute the Calinski-Harabasz index for the top-2 candidate k values and pick the one with the higher index. Confidence: low (reported in the result with a quality warning).

The result always includes both the elected k and the individual method recommendations, so callers can inspect the decision and override it if desired.

### TF-IDF Topic Labeling

When items are provided with associated text, each cluster is labeled with a set of keywords that best distinguish it from other clusters. The labeling algorithm uses TF-IDF (term frequency -- inverse document frequency) weighting at the cluster level, treating each cluster as a "document" and the entire collection of clusters as the "corpus."

For a term t in cluster c:
```
TF(t, c)  = count of t in all texts in cluster c / total terms in cluster c
IDF(t)    = log(num_clusters / num_clusters_containing_t)
TF-IDF(t, c) = TF(t, c) * IDF(t)
```

Terms with high TF-IDF in a cluster are frequent within that cluster but rare across other clusters -- they are the terms that most specifically characterize the cluster's topic.

The top-N TF-IDF terms per cluster form the cluster's label (default N = 5). The label is returned as an array of `{ term, score }` objects and as a human-readable string (the top 3 terms joined by ", "). Stop words (common English function words) are filtered out before scoring. Terms are lowercased and punctuation-stripped.

### k-means++ Initialization

Standard random initialization (Forgy method -- randomly select k points as initial centroids) is simple but can produce poor initializations where all initial centroids are clustered in one region of the space, leading to slow convergence and poor local optima.

k-means++ (Arthur and Vassilvitskii, 2007) provides a probabilistic initialization that spreads the initial centroids across the space:

1. Choose the first centroid uniformly at random from the data points.
2. For each remaining data point x, compute D(x) = the squared distance from x to the nearest already-chosen centroid.
3. Choose the next centroid from the data points with probability proportional to D(x)^2.
4. Repeat steps 2-3 until k centroids are chosen.

This D^2-weighted selection ensures the initial centroids are spread out, dramatically reducing the probability of landing in a poor local optimum. k-means++ has a provably better expected approximation ratio (O(log k)) compared to random initialization. Empirically, k-means++ converges in fewer iterations and produces lower inertia on average, at the cost of O(n * k) work for initialization (negligible compared to the iteration cost for typical n and k).

`embed-cluster` uses k-means++ as the default and only initialization strategy. The Forgy (purely random) initialization is available as an option for reproducibility testing but is not recommended for production use.

---

## 5. Clustering Algorithm

### Lloyd's Algorithm (Standard k-means)

`embed-cluster` implements Lloyd's algorithm with k-means++ initialization and cosine distance. The algorithm operates in three phases.

#### Phase 1: Initialization (k-means++)

```
centroids = []
centroids[0] = data[random_index]   // first centroid chosen uniformly at random

for i = 1 to k-1:
  for each point x in data:
    D[x] = min_{c in centroids} cosine_distance(x, c)^2
  probabilities = D / sum(D)
  centroids[i] = data sampled with probability proportional to probabilities
```

This runs in O(n * k) time for n points and k clusters. For n = 10,000 and k = 20, this is 200,000 distance computations -- fast.

#### Phase 2: Assignment

For each point x in the dataset, assign it to the cluster whose centroid minimizes cosine distance:

```
label[x] = argmin_{i=0..k-1} cosine_distance(x, centroids[i])
```

For cosine distance on L2-normalized vectors, `cosine_distance(x, c) = 1 - dot(x, c)`, so the assignment step reduces to finding the centroid with maximum dot product -- equivalent to a matrix-vector multiply. For n points, k centroids, and d dimensions, this is O(n * k * d) per iteration.

**Vectorized implementation**: The assignment step is implemented as a matrix multiplication: if X is an n×d matrix of (already normalized) data points and C is a k×d matrix of (normalized) centroids, then the cosine similarities of all n points against all k centroids is the matrix product `X @ C.T`, an n×k matrix. The argmax of each row gives the cluster assignment. This formulation allows efficient batch computation using typed arrays.

#### Phase 3: Update (Centroid Recomputation)

For each cluster i, recompute the centroid as the mean of all assigned points:

```
new_centroid[i] = mean({ x : label[x] == i })
// For cosine distance: L2-normalize new_centroid[i] after computing the mean
```

If a cluster becomes empty during an update (all of its points were reassigned to other clusters), the centroid is re-initialized by selecting the data point with the highest inertia contribution (the point farthest from its assigned centroid). This prevents the "empty cluster" degenerate case.

#### Convergence

The algorithm iterates Phase 2 and Phase 3 until one of the following conditions is met:

1. **Assignment convergence**: No point changed its cluster assignment in the last iteration. This is the ideal convergence criterion.
2. **Centroid movement threshold**: The maximum movement of any centroid (measured in cosine distance) between iterations is below `tolerance` (default: 1e-4). This handles near-convergence cases where one or two points oscillate between two equidistant clusters.
3. **Maximum iterations**: The iteration count exceeds `maxIterations` (default: 300). The algorithm terminates with the current best assignment.

Typical convergence for well-separated embedding clusters occurs in 10-30 iterations. 100-200 iterations is common for weakly separated clusters. The 300-iteration limit prevents runaway computation on pathological inputs.

#### Multiple Restarts

A single k-means run may converge to a local optimum rather than the global optimum, depending on the random initialization. To mitigate this, `embed-cluster` runs k-means `nInit` times (default: 10) with different k-means++ random seeds and keeps the run that produced the lowest inertia.

The cost of 10 restarts is 10x the cost of a single run, but since each restart is independent, they can be parallelized using Node.js worker threads for large datasets (opt-in via `parallel: true` option). For the typical dataset (< 5,000 points, < 2048 dimensions, k < 30), 10 sequential restarts complete in under 5 seconds.

#### Computing Inertia

After convergence, inertia for the run is:

```
inertia = Σ_{i=0..k-1} Σ_{x: label[x]==i} cosine_distance(x, centroids[i])^2
```

For unit-normalized vectors, this is equivalent to `Σ (1 - dot(x, centroid[label[x]]))^2`, which can be computed efficiently in a single pass over the assignments.

### Distance Metrics

#### Cosine Distance (Default)

`distanceMetric: 'cosine'` (default). All input vectors are L2-normalized before clustering. The assignment step uses maximum dot product (equivalent to minimum cosine distance). Centroids are re-normalized after each update. This is equivalent to spherical k-means.

Pre-normalization step:
```
for each vector v in data:
  norm = sqrt(dot(v, v))
  if norm > 0:
    v = v / norm
  else:
    v = zero vector (degenerate case, no change)
```

Pre-normalization is done once before the first run and reused across all restarts. The original vectors are not mutated; a normalized copy is maintained internally.

#### Euclidean Distance (Alternative)

`distanceMetric: 'euclidean'`. Standard k-means with Euclidean distance and arithmetic-mean centroids (no re-normalization). Use this when:
- Embedding magnitudes are meaningful (rare for modern LM embeddings, but relevant for some specialized embedding schemes).
- The input dimensionality has been reduced to a small number of dimensions (< 50) via PCA, where Euclidean distance is again discriminative.
- Reproducibility with external tools that use Euclidean k-means is required.

For embeddings produced by standard LM APIs, cosine distance almost always produces better clustering quality (higher silhouette scores).

---

## 6. Automatic Cluster Count Detection

### Why Auto-k Selection Matters

Requiring the user to specify k is the single largest usability barrier for embedding clustering. Unlike image datasets where the number of categories is known (10 classes of handwritten digits, 1000 ImageNet classes), embedding corpora rarely have a known number of topics. A team clustering a document corpus for the first time has no ground truth for k. Getting k wrong by even 2-3 clusters significantly degrades cluster quality: too few clusters produces coarse, mixed-topic groups; too many produces fragmented, near-identical groups.

`embed-cluster` provides fully automatic k selection by default, with the option to specify k exactly when it is known.

### The k Search Range

By default, k is searched in the range `[kMin, kMax]` where `kMin = 2` (a single cluster is trivial -- all points in one group) and `kMax = min(floor(sqrt(n)), 30)` where n is the number of input points. The `sqrt(n)` heuristic comes from the empirical observation that the optimal number of clusters is often around `sqrt(n / 2)` for uniformly distributed data. The 30-cluster cap prevents excessive computation for large datasets.

Both `kMin` and `kMax` are configurable. Setting `kMin = kMax` disables auto-detection and clusters at exactly that k.

### Elbow Method Implementation

The elbow method runs full k-means (with all restarts) for each candidate k and collects the resulting inertia:

```
inertias = {}
for k = kMin to kMax:
  result = kmeans(data, k, nInit=nInit)
  inertias[k] = result.inertia
```

The kneedle algorithm is then applied to the `(k, inertia)` curve:

1. Normalize: `k_norm = (k - kMin) / (kMax - kMin)`, `i_norm = (inertia - min_inertia) / (max_inertia - min_inertia)`.
2. For each k, compute the "difference" from the diagonal: `diff[k] = i_norm[k] - k_norm[k]`. (For a convex-decreasing curve, the diagonal goes from (0, 1) to (1, 0) in normalized space, and the elbow is the k with maximum difference from this diagonal.)
3. The elbow is `argmax(diff)`.

The elbow method's k recommendation is stored as `elbowK`. It is fast (no pairwise distance computation) but can be unreliable when the inertia curve is smooth with no sharp inflection -- a common occurrence when the natural cluster count is near the geometric mean of `kMin` and `kMax`.

### Silhouette Method Implementation

The silhouette method computes the mean silhouette coefficient for each candidate k:

```
silhouetteScores = {}
for k = kMin to kMax:
  result = kmeans(data, k, nInit=nInit)
  silhouetteScores[k] = meanSilhouette(data, result.labels)
```

Computing mean silhouette for a given clustering requires:

1. For each point x in cluster C_i:
   - `a(x)` = mean cosine distance from x to all other points in C_i.
   - For each other cluster C_j (j ≠ i):
     - `d(x, C_j)` = mean cosine distance from x to all points in C_j.
   - `b(x)` = min over j≠i of `d(x, C_j)`.
   - `s(x)` = (b(x) - a(x)) / max(a(x), b(x)).
2. Mean silhouette = mean over all x of s(x).

Silhouette computation is O(n^2) in the number of points, which is expensive for large datasets. For n > 1000, `embed-cluster` uses approximate silhouette via reservoir sampling: for each point, compute a(x) and b(x) using a random sample of `silhouetteSampleSize` (default: 500) points per cluster rather than all points in the cluster. This provides a good approximation in O(n * silhouetteSampleSize) time. For n <= 1000, exact silhouette is always computed.

The silhouette method's k recommendation is `argmax(silhouetteScores)`. It is the more reliable of the two methods because it directly measures clustering quality (cohesion and separation), not just inertia. The tradeoff is O(n^2) work per candidate k, making it slower than the elbow method for large datasets.

### Calinski-Harabasz Index (Tiebreaker)

When the elbow and silhouette methods disagree by 2 or more clusters, the Calinski-Harabasz (CH) index (also called the Variance Ratio Criterion) is computed for the top-2 candidate k values as a tiebreaker.

```
CH(k) = [SSB / (k - 1)] / [SSW / (n - k)]
```

where SSB is the between-cluster sum of squares (sum of squared distances from each cluster centroid to the global centroid, weighted by cluster size) and SSW is the within-cluster sum of squares (total inertia). A higher CH index indicates better-separated clusters relative to their compactness.

The CH index is O(n) to compute (given the clustering), making it cheap as a tiebreaker. The k with the higher CH index is selected, and the result is flagged with `confidence: 'low'` to indicate that the auto-detection methods disagreed.

### Combined Auto-K Result

The `findOptimalK` function returns:

```typescript
interface OptimalKResult {
  k: number;              // elected optimal k
  confidence: 'high' | 'medium' | 'low';
  elbowK: number;         // elbow method's recommendation
  silhouetteK: number;    // silhouette method's recommendation
  tiebreakerK?: number;   // CH index tiebreaker result, if used
  elbowInertias: Record<number, number>;     // inertia for each k tried
  silhouetteScores: Record<number, number>; // mean silhouette for each k tried
}
```

### User-Specified k

When `k` is provided explicitly in `ClusterOptions`, the auto-detection phase is skipped entirely. The algorithm runs k-means at the specified k with all restarts and returns the result. This is appropriate when:
- The desired number of clusters is known from domain knowledge.
- The caller is running a sweep and wants to test specific k values.
- Speed is critical and auto-detection's O(n^2 * (kMax - kMin)) cost is unacceptable.

---

## 7. Cluster Quality Metrics

### Per-Point Silhouette Coefficient

Every point in every cluster has a silhouette coefficient `s(x)` in [-1, 1] (see Section 4 for the formula). `embed-cluster` computes and returns the per-point silhouette coefficient in the `ClusterItem.silhouette` field of every item in every cluster. This enables:

- Identifying the "best member" of a cluster: the point with the highest silhouette score is the most centrally located within its cluster, making it a good representative for display or summarization.
- Identifying "border members": points with silhouette near 0 are on the boundary between two clusters. They may be equally valid members of either.
- Identifying outliers: points with negative silhouette are likely misassigned. They are closer to a neighboring cluster than their own.

### Per-Cluster Silhouette Score

The per-cluster silhouette score is the mean silhouette coefficient of all points in the cluster:

```
clusterSilhouette[i] = mean({ s(x) : label[x] == i })
```

A cluster with high silhouette score is compact and well-separated from its neighbors. A cluster with low or negative silhouette score is diffuse or overlapping with adjacent clusters. Per-cluster scores allow identifying which clusters are "good" (well-separated) vs. "bad" (merged or fragmented) even when the overall mean score is acceptable.

### Inertia (Per-Cluster and Total)

Per-cluster inertia is the within-cluster sum of squared cosine distances to the centroid. The total inertia is the sum of per-cluster inertias. These are returned in `Cluster.inertia` and `ClusterResult.inertia`.

### Inter-Cluster Distances

The pairwise cosine distance matrix between all cluster centroids is computed and returned in `ClusterResult.interClusterDistances`. This is a k×k symmetric matrix where `interClusterDistances[i][j]` is the cosine distance between centroids i and j. The minimum off-diagonal entry indicates the closest pair of clusters -- a small value may indicate over-clustering (two clusters that should be merged). The maximum entry indicates the most semantically distinct cluster pair.

The "separation" summary metric is the mean of all off-diagonal entries, representing the average distance between cluster centroids. Higher separation is better -- it means the clusters are spread across the embedding space.

### Cluster Size Distribution

The `quality.sizeVariance` field reports the variance in cluster sizes (number of members). Ideally, clusters should be roughly equal in size. Very unequal cluster sizes may indicate:
- One "catch-all" cluster that captures heterogeneous content.
- A small cluster of very specific, tightly-grouped content.
- An over-clustered setting where the majority cluster was unnecessarily split.

The coefficient of variation (standard deviation / mean) is reported as `quality.sizeCV`. A CV below 0.5 indicates roughly balanced clusters; above 1.0 indicates high size imbalance.

### Outlier Detection

Points with a negative silhouette coefficient are tagged as outliers. The `ClusterResult.outliers` array contains references to all items with `silhouette < silhouetteOutlierThreshold` (default: 0). Outliers are still assigned to their best-fit cluster but are flagged in the result. The `quality.outlierRate` field reports the fraction of all points that are outliers.

A high outlier rate may indicate:
- The data has a significant proportion of noise or off-topic content.
- The selected k is too low (some distinct groups are merged, causing many points to be near cluster boundaries).
- The data does not have a clean cluster structure (inherently high-dimensional noise).

### Overall Quality Report

The `ClusterResult.quality` object aggregates all quality signals:

```typescript
interface ClusterQuality {
  meanSilhouette: number;       // mean silhouette score across all points [-1, 1]
  minClusterSilhouette: number; // lowest per-cluster silhouette score
  maxClusterSilhouette: number; // highest per-cluster silhouette score
  totalInertia: number;         // total within-cluster sum of squared distances
  meanInterClusterDistance: number; // mean pairwise centroid distance
  minInterClusterDistance: number;  // closest centroid pair distance (separation risk)
  outlierRate: number;          // fraction of points with negative silhouette [0, 1]
  sizeCV: number;               // coefficient of variation of cluster sizes
  interpretation: 'excellent' | 'good' | 'fair' | 'poor'; // overall quality band
}
```

The `interpretation` field applies a simple rubric:
- **excellent**: meanSilhouette >= 0.7 and outlierRate <= 0.05.
- **good**: meanSilhouette >= 0.5 and outlierRate <= 0.15.
- **fair**: meanSilhouette >= 0.25 and outlierRate <= 0.30.
- **poor**: meanSilhouette < 0.25 or outlierRate > 0.30.

---

## 8. Topic Labeling

### When Labels Are Generated

Topic labels are generated when the input contains text alongside embeddings (i.e., when `EmbedItem[]` is passed rather than `number[][]`). Labels are generated after clustering is complete, as a post-processing step. The labeling algorithm does not influence which cluster a point is assigned to.

### TF-IDF Keyword Extraction

The TF-IDF labeler treats each cluster's combined texts as a "document" in a corpus of k "documents."

**Tokenization**: Each text is split into tokens by lowercasing, stripping punctuation, and splitting on whitespace. Tokens shorter than 3 characters are discarded. A built-in stop word list of ~200 common English function words (the, is, a, of, and, to, in, that, for, it, are, this, was, be, as, etc.) is applied to eliminate noise terms.

**TF computation** (term frequency within a cluster):

```
TF(term, cluster) = count(term in cluster) / total_tokens_in_cluster
```

**IDF computation** (inverse document frequency across clusters):

```
IDF(term) = log((1 + k) / (1 + count(clusters containing term))) + 1
```

The "+1" smoothing in numerator and denominator prevents zero-IDF for terms that appear in all clusters and handles the single-cluster edge case.

**TF-IDF score**:

```
score(term, cluster) = TF(term, cluster) * IDF(term)
```

The top `labelTopN` terms by TF-IDF score for each cluster are returned as the cluster's keyword label. The default `labelTopN = 5`. The human-readable label string is the top 3 terms joined by ", " (e.g., `"kubernetes, deployment, container"`).

**Bigram extraction**: In addition to unigrams, the labeler extracts bigrams (two-word sequences) from each text and scores them using the same TF-IDF formula. Bigrams that score above a threshold are preferred over unigrams in the label, as they often provide more specific topic descriptions (e.g., `"machine learning"` is more informative than `"machine"` or `"learning"` separately). Bigrams are formed from adjacent non-stop-word tokens.

### Centroid-Nearest Representative

In addition to TF-IDF keywords, `embed-cluster` identifies the "most representative" item in each cluster: the item whose embedding vector is closest to the cluster centroid in cosine distance. This item is stored in `Cluster.representative`. It serves as a concrete example of the cluster's topic, useful for display ("this cluster is about: [representative text]") and for manual label verification.

The representative is computed by:

```
representative = argmin_{x in cluster_i} cosine_distance(x.embedding, centroid[i])
```

This is O(n_i * d) per cluster, where n_i is the cluster size and d is the embedding dimension.

### Custom Labeler

The built-in TF-IDF labeler works well for document corpora but may not produce meaningful labels for:
- Very short texts (tweets, titles, single sentences) where term frequency is insufficient.
- Technical corpora with high domain jargon not in the default stop word list.
- Non-English corpora.

For these cases, callers provide a `labeler` function:

```typescript
type LabelerFn = (cluster: {
  items: EmbedItem[];
  centroid: number[];
  representative: EmbedItem;
}) => Promise<string> | string;
```

The labeler receives the full cluster with all its member items, the centroid vector, and the representative item. It can call an LLM, apply a domain-specific taxonomy, or implement any custom logic. The returned string becomes `Cluster.label`.

**LLM labeling example**:

```typescript
const result = await cluster(items, {
  labeler: async ({ items, representative }) => {
    const texts = items.slice(0, 10).map(i => i.text).join('\n---\n');
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{
        role: 'user',
        content: `Give a 3-5 word topic label for these texts:\n${texts}`
      }],
    });
    return response.choices[0].message.content?.trim() ?? 'Unlabeled';
  },
});
```

Labeler functions are called concurrently for all clusters (with `Promise.all`), minimizing wall-clock time for API-backed labelers.

### Label Quality

`embed-cluster` computes a `labelConfidence` score per cluster that estimates how well the TF-IDF keywords describe the cluster. It is computed as the normalized TF-IDF score of the top keyword relative to the maximum TF-IDF score observed across all clusters. A cluster with high TF-IDF scores for its top terms has distinctive vocabulary -- those terms genuinely characterize the cluster. A cluster with uniformly low TF-IDF scores has no distinctive vocabulary (all its terms appear equally in all clusters), indicating the cluster is not semantically well-defined.

`labelConfidence` ranges from 0 to 1. Values below 0.2 indicate a cluster with weak topical coherence that may benefit from re-labeling with a custom labeler or manual review.

---

## 9. API Surface

### Core Functions

#### `cluster(input, options?)`

The primary API. Accepts either raw embeddings or structured items.

```typescript
// Overload 1: raw embeddings only (no text, no TF-IDF labels)
function cluster(
  embeddings: number[][],
  options?: ClusterOptions,
): Promise<ClusterResult>;

// Overload 2: items with text and embeddings (enables TF-IDF labeling)
function cluster(
  items: EmbedItem[],
  options?: ClusterOptions,
): Promise<ClusterResult>;
```

**Behavior**: Validates input, optionally runs PCA preprocessing, auto-detects k (or uses provided k), runs k-means with restarts, computes silhouette scores, generates labels, and returns `ClusterResult`.

**Throws** `ClusterError` if:
- Input array is empty or contains fewer than `kMin + 1` points.
- All embedding vectors are zero vectors (degenerate input).
- Embedding vectors have inconsistent dimensionality.
- `kMin >= kMax` when both are specified (only when k is auto-detected).

#### `findOptimalK(embeddings, options?)`

Runs only the k-detection phase without clustering the full dataset.

```typescript
function findOptimalK(
  embeddings: number[][],
  options?: {
    kMin?: number;          // default: 2
    kMax?: number;          // default: min(floor(sqrt(n)), 30)
    nInit?: number;         // k-means restarts per k candidate, default: 3
    distanceMetric?: 'cosine' | 'euclidean'; // default: 'cosine'
    silhouetteSampleSize?: number;           // default: 500
    methods?: Array<'elbow' | 'silhouette'>; // which methods to run, default: both
  },
): Promise<OptimalKResult>;
```

Useful when the caller wants to inspect k-selection behavior before committing to a full clustering run.

#### `silhouetteScore(embeddings, labels)`

Computes silhouette coefficients for an existing clustering assignment.

```typescript
function silhouetteScore(
  embeddings: number[][],
  labels: number[],  // cluster assignment for each embedding (0-indexed)
  options?: {
    distanceMetric?: 'cosine' | 'euclidean'; // default: 'cosine'
    sampleSize?: number;                     // default: exact for n<=1000, 500 otherwise
  },
): Promise<SilhouetteResult>;
```

Allows callers to evaluate any external clustering (from `ml-kmeans`, a vector database query, or a manual assignment) using `embed-cluster`'s silhouette implementation.

#### `createClusterer(config)`

Factory for a configured clusterer instance. Useful when the same configuration is applied to many clustering runs.

```typescript
function createClusterer(config: ClusterOptions): Clusterer;

interface Clusterer {
  cluster(embeddings: number[][]): Promise<ClusterResult>;
  cluster(items: EmbedItem[]): Promise<ClusterResult>;
  findOptimalK(embeddings: number[][]): Promise<OptimalKResult>;
}
```

### Type Definitions

#### Input Types

```typescript
/** A single item to be clustered: text paired with its embedding vector. */
interface EmbedItem {
  /** The text this embedding represents. Used for TF-IDF topic labeling. */
  text: string;

  /** The embedding vector. Must have the same dimensionality as all other items. */
  embedding: number[];

  /**
   * Optional identifier. If provided, appears in ClusterItem.id.
   * If not provided, a sequential index is used.
   */
  id?: string;

  /**
   * Optional metadata record for arbitrary caller-defined properties
   * (e.g., document source URL, creation timestamp, category).
   * Passed through to ClusterItem.metadata without modification.
   */
  metadata?: Record<string, unknown>;
}
```

#### Options

```typescript
interface ClusterOptions {
  /**
   * Exact number of clusters. If specified, auto-detection is skipped.
   * Must be >= 2 and <= the number of input points.
   * Default: auto-detected.
   */
  k?: number;

  /**
   * Minimum k to consider during auto-detection.
   * Default: 2.
   */
  kMin?: number;

  /**
   * Maximum k to consider during auto-detection.
   * Default: min(floor(sqrt(n)), 30).
   */
  kMax?: number;

  /**
   * Distance metric for k-means and silhouette computation.
   * - 'cosine': cosine distance (1 - cosine_similarity). Default. Recommended for LM embeddings.
   * - 'euclidean': Euclidean distance. Better after PCA to low dimensions.
   * Default: 'cosine'.
   */
  distanceMetric?: 'cosine' | 'euclidean';

  /**
   * Number of k-means restarts per k candidate (and for the final clustering).
   * The run with the lowest inertia is kept.
   * Default: 10.
   */
  nInit?: number;

  /**
   * Maximum number of iterations per k-means run.
   * Default: 300.
   */
  maxIterations?: number;

  /**
   * Convergence tolerance: maximum centroid movement (cosine distance)
   * to consider the algorithm converged.
   * Default: 1e-4.
   */
  tolerance?: number;

  /**
   * Random seed for reproducible results. When provided, the same seed
   * produces the same clustering given the same input and options.
   * Default: random (non-reproducible).
   */
  seed?: number;

  /**
   * Auto-detection methods to run. Relevant only when k is not specified.
   * Default: ['elbow', 'silhouette'].
   */
  autoKMethods?: Array<'elbow' | 'silhouette'>;

  /**
   * Number of points per cluster to sample for approximate silhouette
   * computation. For n > 1000, exact silhouette is O(n^2); sampling
   * provides a fast approximation.
   * Default: 500. Set to Infinity to always compute exact silhouette.
   */
  silhouetteSampleSize?: number;

  /**
   * Points with silhouette below this threshold are flagged as outliers
   * in ClusterResult.outliers.
   * Default: 0 (all points with negative silhouette).
   */
  silhouetteOutlierThreshold?: number;

  /**
   * PCA preprocessing. When enabled, input vectors are projected to a
   * lower-dimensional space before clustering. Speeds up clustering for
   * very high-dimensional inputs (>1000 dimensions) and can improve
   * cluster quality by removing noise dimensions.
   * Default: false.
   */
  pca?: {
    enabled: boolean;
    /** Target dimensions. Default: 50. */
    dimensions?: number;
  };

  /**
   * Topic labeling configuration. Only used when EmbedItem[] is provided.
   */
  labeling?: {
    /**
     * Number of top TF-IDF terms to extract per cluster.
     * Default: 5.
     */
    topN?: number;

    /**
     * Custom labeler function. When provided, replaces TF-IDF labeling.
     * Called once per cluster after clustering completes.
     * Default: built-in TF-IDF labeler.
     */
    labeler?: LabelerFn;

    /**
     * Include bigrams in TF-IDF extraction.
     * Default: true.
     */
    includeBigrams?: boolean;

    /**
     * Additional stop words to filter beyond the built-in English list.
     * Default: [].
     */
    additionalStopWords?: string[];
  };

  /**
   * Whether to compute 2D PCA projections for visualization export.
   * Adds a { x, y } field to each ClusterItem.
   * Default: false.
   */
  computeVisualization?: boolean;

  /**
   * Whether to run k-means restarts in parallel using Node.js worker threads.
   * Only beneficial for large datasets (>5000 points) or high nInit values.
   * Default: false.
   */
  parallel?: boolean;
}
```

#### Result Types

```typescript
/** The full result of a clustering operation. */
interface ClusterResult {
  /** The number of clusters found. */
  k: number;

  /** All clusters. */
  clusters: Cluster[];

  /** Overall quality metrics for this clustering. */
  quality: ClusterQuality;

  /** Total within-cluster sum of squared distances (inertia). */
  inertia: number;

  /** Mean silhouette score across all points [-1, 1]. */
  silhouetteScore: number;

  /** k×k matrix of pairwise cosine distances between cluster centroids. */
  interClusterDistances: number[][];

  /**
   * All items with negative silhouette (silhouette < outlierThreshold).
   * These points are still assigned to a cluster but are flagged as
   * potentially misassigned.
   */
  outliers: ClusterItem[];

  /**
   * Auto-detection result. Undefined when k was specified by the caller.
   */
  autoK?: OptimalKResult;

  /**
   * 2D visualization data for all points.
   * Populated only when computeVisualization: true.
   */
  visualization?: VisualizationData;
}

/** A single cluster. */
interface Cluster {
  /** Zero-based cluster index. */
  id: number;

  /**
   * Human-readable topic label.
   * Populated only when EmbedItem[] input was provided.
   */
  label?: string;

  /**
   * Top TF-IDF keywords for this cluster with their scores.
   * Populated only when EmbedItem[] input was provided and labeling.labeler is not set.
   */
  keywords?: Array<{ term: string; score: number }>;

  /**
   * Confidence score for the topic label [0, 1].
   * Higher means the label terms are more distinctive to this cluster.
   */
  labelConfidence?: number;

  /** The centroid embedding vector. Length equals input embedding dimensionality. */
  centroid: number[];

  /** All items assigned to this cluster. */
  items: ClusterItem[];

  /** Number of items in this cluster. */
  size: number;

  /**
   * The item whose embedding is closest to the centroid.
   * The most representative member of the cluster.
   */
  representative?: ClusterItem;

  /** Within-cluster sum of squared distances to centroid (inertia). */
  inertia: number;

  /** Mean silhouette score for all points in this cluster. */
  silhouetteScore: number;
}

/** A single item within a cluster. */
interface ClusterItem {
  /** The cluster ID this item belongs to (matches Cluster.id). */
  clusterId: number;

  /** Caller-provided ID, or sequential index if not provided. */
  id: string;

  /** The original embedding vector (before any normalization). */
  embedding: number[];

  /** The text, if EmbedItem[] was the input. */
  text?: string;

  /** Caller-provided metadata, passed through unchanged. */
  metadata?: Record<string, unknown>;

  /** Silhouette coefficient for this item [-1, 1]. */
  silhouette: number;

  /** Cosine distance from this item's centroid. */
  distanceToCentroid: number;

  /**
   * 2D PCA projection coordinates for visualization.
   * Populated only when computeVisualization: true.
   */
  projection?: { x: number; y: number };
}

/** Result of silhouette score computation for an existing clustering. */
interface SilhouetteResult {
  /** Mean silhouette coefficient across all points. */
  meanScore: number;

  /** Per-point silhouette coefficients, in input order. */
  scores: number[];

  /** Per-cluster mean silhouette scores. */
  clusterScores: Record<number, number>;

  /** Points with silhouette < 0 (potential misassignments). */
  outlierIndices: number[];
}

/** Result of automatic k selection. */
interface OptimalKResult {
  /** The elected optimal k. */
  k: number;

  /** Confidence in the election. */
  confidence: 'high' | 'medium' | 'low';

  /** Elbow method's k recommendation. */
  elbowK: number;

  /** Silhouette method's k recommendation. */
  silhouetteK: number;

  /** CH index tiebreaker result, populated when elbow and silhouette disagreed by >= 2. */
  tiebreakerK?: number;

  /** Inertia at each k tried. */
  elbowInertias: Record<number, number>;

  /** Mean silhouette score at each k tried. */
  silhouetteScores: Record<number, number>;
}

/** Overall cluster quality summary. */
interface ClusterQuality {
  meanSilhouette: number;
  minClusterSilhouette: number;
  maxClusterSilhouette: number;
  totalInertia: number;
  meanInterClusterDistance: number;
  minInterClusterDistance: number;
  outlierRate: number;
  sizeCV: number;
  interpretation: 'excellent' | 'good' | 'fair' | 'poor';
}

/** Visualization export data for d3, plotly, or Observable. */
interface VisualizationData {
  /**
   * Per-point 2D PCA coordinates and cluster assignment.
   * In input order.
   */
  points: Array<{
    x: number;
    y: number;
    clusterId: number;
    id: string;
    text?: string;
  }>;

  /**
   * Cluster centroid 2D PCA coordinates.
   */
  centroids: Array<{
    x: number;
    y: number;
    clusterId: number;
    label?: string;
  }>;

  /**
   * Fraction of total variance explained by the 2D projection.
   * 1.0 means the 2D projection captures all variance; lower values
   * mean some structure is lost in the projection.
   */
  varianceExplained: number;
}

/** Labeler function type for custom cluster label generation. */
type LabelerFn = (cluster: {
  items: EmbedItem[];
  centroid: number[];
  representative: EmbedItem;
  id: number;
}) => Promise<string> | string;

/** Error thrown by embed-cluster on invalid input or configuration. */
class ClusterError extends Error {
  code: 'EMPTY_INPUT' | 'INCONSISTENT_DIMENSIONS' | 'DEGENERATE_INPUT'
       | 'INVALID_K' | 'INVALID_OPTIONS';
}
```

### Usage Examples

#### Basic Clustering with Auto-k

```typescript
import { cluster } from 'embed-cluster';

// Assume embeddings is number[][] from your embedding API
const result = await cluster(embeddings);

console.log(`Found ${result.k} clusters`);
console.log(`Mean silhouette: ${result.silhouetteScore.toFixed(3)}`);
console.log(`Quality: ${result.quality.interpretation}`);

for (const c of result.clusters) {
  console.log(`Cluster ${c.id}: ${c.size} items, silhouette=${c.silhouetteScore.toFixed(3)}`);
}
```

#### Clustering with Text Items and TF-IDF Labels

```typescript
import { cluster } from 'embed-cluster';

const items = documents.map((doc, i) => ({
  id: doc.id,
  text: doc.content,
  embedding: doc.embedding,
  metadata: { source: doc.url },
}));

const result = await cluster(items, { kMin: 3, kMax: 20 });

for (const c of result.clusters) {
  console.log(`\nCluster: ${c.label}`);
  console.log(`  Size: ${c.size} documents`);
  console.log(`  Quality: ${c.silhouetteScore.toFixed(3)}`);
  console.log(`  Top terms: ${c.keywords?.map(k => k.term).join(', ')}`);
  console.log(`  Representative: ${c.representative?.text?.slice(0, 80)}...`);
}
```

#### Fixed-k Clustering

```typescript
import { cluster } from 'embed-cluster';

// Group search results into exactly 5 topic facets
const result = await cluster(searchResults.map(r => ({
  text: r.snippet,
  embedding: r.embedding,
  id: r.id,
})), { k: 5 });
```

#### LLM-Powered Labels

```typescript
import { cluster } from 'embed-cluster';
import OpenAI from 'openai';

const openai = new OpenAI();

const result = await cluster(items, {
  labeling: {
    labeler: async ({ items, representative }) => {
      const samples = items.slice(0, 5).map(i => `- ${i.text}`).join('\n');
      const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content:
          `Give a concise 3-5 word topic label for this group of texts:\n${samples}\n\nLabel:` }],
        max_tokens: 20,
      });
      return response.choices[0].message.content?.trim() ?? 'Miscellaneous';
    },
  },
});
```

#### Evaluating an Existing Clustering

```typescript
import { silhouetteScore } from 'embed-cluster';

// labels[i] is the cluster assignment (0-indexed) for embeddings[i]
const result = await silhouetteScore(embeddings, labels);

console.log(`Mean silhouette: ${result.meanScore.toFixed(3)}`);
console.log(`Outlier count: ${result.outlierIndices.length}`);
```

#### Visualization Export

```typescript
import { cluster } from 'embed-cluster';

const result = await cluster(items, { computeVisualization: true });

// result.visualization.points contains { x, y, clusterId, id, text } for each item
// result.visualization.centroids contains { x, y, clusterId, label } for each centroid
// Ready for d3.js, plotly.js, or Observable notebooks

const vizJson = JSON.stringify(result.visualization, null, 2);
fs.writeFileSync('cluster-viz.json', vizJson);
```

---

## 10. Dimensionality Reduction

### Why Dimensionality Reduction for Embeddings

Modern LM embeddings have 768 to 3072 dimensions. Clustering in 3072 dimensions is mathematically valid but computationally expensive: every distance computation is a 3072-element dot product. For a dataset of n = 5000 points and k = 20 clusters, each k-means iteration involves 5000 * 20 = 100,000 distance computations, each requiring 3072 multiplications and additions. With 30 iterations and 10 restarts, this is 900 million floating-point operations per k candidate -- and with auto-k scanning k = 2 through 30, the total is 27 billion operations.

Dimensionality reduction via PCA projects the embeddings from d dimensions to d' << d dimensions while preserving the largest axes of variance. For d = 1536 and d' = 50, each distance computation is 30x cheaper. The speed improvement is roughly linear in the reduction ratio.

More subtly, very high-dimensional embeddings often contain many "noise" dimensions that carry little semantic information and add Euclidean/cosine distance noise. Projecting to the top-50 principal components (which capture the dominant variance axes) can improve cluster separation by reducing this noise. This is not always the case -- for very large, diverse corpora, the top-50 components may not capture enough variance -- which is why PCA preprocessing is opt-in rather than default.

### PCA Implementation

`embed-cluster` includes a built-in lightweight PCA implementation using the power iteration method for extracting the top-k principal components. This avoids a dependency on `ml-pca` for the common case of needing fewer than 100 components.

The power iteration PCA:

1. Center the data: subtract the mean vector from each point.
2. For each component i = 1 to d':
   a. Initialize a random unit vector v.
   b. Repeat until convergence: v' = X.T @ (X @ v), normalize v' to unit length.
   c. The converged v is the i-th principal component.
   d. Deflate: X = X - (X @ v) outer (v) to remove the variance explained by this component.
3. Project the original data: X' = X_centered @ V (where V is the d×d' matrix of principal components).

For very large datasets (n > 50,000) or very high target dimensions (d' > 100), the built-in implementation is slow. The `ml-pca` package can be used as a peer dependency for faster SVD-based PCA in these cases.

### When to Enable PCA

| Scenario | PCA Recommended? |
|----------|-----------------|
| d > 1000, n < 50,000, d' = 50 | Yes -- significant speedup, usually no quality loss |
| d = 1536, n < 5,000 | Optional -- modest speedup |
| d = 1536, n < 1,000 | No -- overhead not worth it |
| d < 300 | No -- already manageable dimensionality |
| Visualization desired | Always -- PCA to d'=2 is used for visualization regardless |

### PCA vs. Input Dimensionality

When PCA is enabled, the internal clustering operates on PCA-projected vectors. The output centroids are always reported in the original embedding space (by projecting the low-dimensional centroids back to full dimensionality). This ensures that `Cluster.centroid` can be used for similarity search in the original embedding space (e.g., querying a vector database with a cluster centroid).

### Variance Explained

The `VisualizationData.varianceExplained` field reports the fraction of total variance captured by the 2D PCA projection. Values above 0.5 indicate the 2D projection is reasonably faithful. Values below 0.2 indicate significant structure is lost in the projection -- the 2D plot may visually mis-represent the true cluster separation.

---

## 11. Visualization Export

### Purpose

`embed-cluster` can export 2D PCA projections of all cluster members and centroids for plotting with external tools. This is a convenience for exploration and debugging -- not a built-in visualization renderer.

### Enabling Visualization

Set `computeVisualization: true` in `ClusterOptions`. This adds a `visualization` field to `ClusterResult` and a `projection: { x, y }` field to each `ClusterItem`.

### Output Format

The `VisualizationData` object is designed for direct consumption by JSON-based visualization libraries:

```json
{
  "points": [
    { "x": -0.82, "y": 0.34, "clusterId": 0, "id": "doc-001", "text": "kubernetes deployment..." },
    { "x": -0.75, "y": 0.41, "clusterId": 0, "id": "doc-002", "text": "container orchestration..." },
    { "x":  1.23, "y": -0.67, "clusterId": 1, "id": "doc-003", "text": "oauth token refresh..." }
  ],
  "centroids": [
    { "x": -0.78, "y": 0.37, "clusterId": 0, "label": "kubernetes, deployment, container" },
    { "x":  1.20, "y": -0.65, "clusterId": 1, "label": "oauth, authentication, token" }
  ],
  "varianceExplained": 0.43
}
```

### Plotly.js Integration

```javascript
const viz = clusterResult.visualization;
const colors = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f'];

const traces = [...new Set(viz.points.map(p => p.clusterId))].map(cId => ({
  x: viz.points.filter(p => p.clusterId === cId).map(p => p.x),
  y: viz.points.filter(p => p.clusterId === cId).map(p => p.y),
  mode: 'markers',
  type: 'scatter',
  name: viz.centroids.find(c => c.clusterId === cId)?.label ?? `Cluster ${cId}`,
  marker: { color: colors[cId % colors.length], size: 6 },
}));

Plotly.newPlot('chart', traces);
```

### d3.js Integration

The `points` array is compatible with d3's data join pattern. Assign point color by `clusterId` and position by `{ x, y }`. The `centroids` array provides centroid positions for label overlays.

---

## 12. CLI

### Installation and Invocation

```bash
# Global install
npm install -g embed-cluster
embed-cluster --input embeddings.json

# npx (no install)
npx embed-cluster --input embeddings.json --output result.json

# Package script
# package.json: { "scripts": { "cluster": "embed-cluster --input embeddings.json" } }
npm run cluster
```

### Input Format

The CLI reads a JSON file containing either:

1. **Array of embedding vectors** (`number[][]`):
```json
[[0.1, 0.2, -0.3, ...], [0.4, -0.1, 0.7, ...], ...]
```

2. **Array of EmbedItem objects**:
```json
[
  { "id": "doc-1", "text": "kubernetes deployment guide", "embedding": [0.1, 0.2, ...] },
  { "id": "doc-2", "text": "container orchestration", "embedding": [0.4, -0.1, ...] }
]
```

### CLI Flags

```
embed-cluster [options]

Input/Output:
  --input <path>          Path to JSON file containing embeddings or EmbedItem[]. Required.
  --output <path>         Write ClusterResult JSON to this file. Default: stdout.
  --format <format>       Output format: json, summary. Default: summary.

Clustering Options:
  --k <n>                 Exact number of clusters. Skips auto-detection.
  --k-min <n>             Minimum k for auto-detection. Default: 2.
  --k-max <n>             Maximum k for auto-detection. Default: sqrt(n) capped at 30.
  --metric <m>            Distance metric: cosine, euclidean. Default: cosine.
  --n-init <n>            k-means restarts. Default: 10.
  --max-iterations <n>    Max iterations per k-means run. Default: 300.
  --seed <n>              Random seed for reproducibility.

Auto-k Options:
  --methods <m>           Auto-k methods: elbow, silhouette, both. Default: both.
  --silhouette-sample <n> Silhouette sample size. Default: 500.

Labeling Options (requires EmbedItem input):
  --top-n <n>             Top TF-IDF terms per label. Default: 5.
  --no-bigrams            Disable bigram extraction.

PCA Options:
  --pca                   Enable PCA preprocessing.
  --pca-dims <n>          PCA target dimensions. Default: 50.

Visualization:
  --viz <path>            Export visualization JSON to this path.

Meta:
  --version               Print version and exit.
  --help                  Print help and exit.
```

### Output Formats

**`--format summary`** (default): Human-readable terminal output.

```
$ embed-cluster --input docs.json

  embed-cluster v0.1.0

  Input:   1247 items, 1536 dimensions
  Method:  cosine k-means++, auto-k (elbow + silhouette)

  Auto-k results:
    Elbow method:       k = 7 (high confidence)
    Silhouette method:  k = 7 (mean score: 0.612)
    Elected k:          7 (confidence: high)

  Clustering results (k=7):
    Mean silhouette:    0.612 (good)
    Total inertia:      142.3
    Outlier rate:       4.1% (51 items)

  Clusters:
    #0  [213 items]  sil=0.71  "kubernetes, deployment, container"
    #1  [198 items]  sil=0.65  "oauth, authentication, token"
    #2  [187 items]  sil=0.59  "database, migration, schema"
    #3  [164 items]  sil=0.68  "monitoring, alerting, prometheus"
    #4  [159 items]  sil=0.57  "python, testing, pytest"
    #5  [141 items]  sil=0.63  "terraform, infrastructure, aws"
    #6  [ 85 items]  sil=0.42  "api, rest, endpoint"

  Outliers: 51 items flagged (silhouette < 0)
  Quality interpretation: good
```

**`--format json`**: Writes the full `ClusterResult` as JSON to stdout or `--output`.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0`  | Clustering completed successfully. |
| `1`  | Input error (file not found, invalid JSON, inconsistent dimensions). |
| `2`  | Configuration error (invalid flags, incompatible options). |
| `3`  | Clustering warning (auto-k detection had low confidence, quality is 'poor'). |

---

## 13. Configuration Reference

### All Options with Defaults

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `k` | `number` | undefined | Exact cluster count. If set, skips auto-detection. |
| `kMin` | `number` | `2` | Minimum k for auto-detection. |
| `kMax` | `number` | `min(floor(sqrt(n)), 30)` | Maximum k for auto-detection. |
| `distanceMetric` | `'cosine' \| 'euclidean'` | `'cosine'` | Distance metric. |
| `nInit` | `number` | `10` | k-means restarts per k candidate. |
| `maxIterations` | `number` | `300` | Max iterations per k-means run. |
| `tolerance` | `number` | `1e-4` | Centroid movement convergence threshold. |
| `seed` | `number` | random | Random seed for reproducibility. |
| `autoKMethods` | `Array<'elbow' \| 'silhouette'>` | `['elbow', 'silhouette']` | Auto-k detection methods. |
| `silhouetteSampleSize` | `number` | `500` | Points sampled per cluster for approximate silhouette. |
| `silhouetteOutlierThreshold` | `number` | `0` | Silhouette below this → flagged as outlier. |
| `pca.enabled` | `boolean` | `false` | Enable PCA preprocessing. |
| `pca.dimensions` | `number` | `50` | PCA target dimensions. |
| `labeling.topN` | `number` | `5` | Top TF-IDF terms per cluster. |
| `labeling.labeler` | `LabelerFn` | TF-IDF built-in | Custom label generator. |
| `labeling.includeBigrams` | `boolean` | `true` | Include bigrams in TF-IDF. |
| `labeling.additionalStopWords` | `string[]` | `[]` | Extra stop words to filter. |
| `computeVisualization` | `boolean` | `false` | Compute 2D PCA projections. |
| `parallel` | `boolean` | `false` | Parallel restarts via worker threads. |

---

## 14. Integration with the Monorepo

### `embed-cache` Integration

`embed-cluster` consumes embedding vectors; `embed-cache` produces them. The typical integration:

```typescript
import { createCache } from 'embed-cache';
import { cluster } from 'embed-cluster';

const cache = createCache({ embedder: openaiEmbed, model: 'text-embedding-3-small' });

const items = documents.map(doc => ({ id: doc.id, text: doc.content }));
const embeddings = await cache.embedBatch(items.map(i => i.text));

const embedItems = items.map((item, i) => ({
  ...item,
  embedding: embeddings[i],
}));

const result = await cluster(embedItems);
```

`embed-cache` ensures embeddings are not recomputed across clustering runs -- useful when experimenting with different k values or options on the same corpus.

### `embed-drift` Integration

`embed-drift` detects distribution shift in embedding streams over time. `embed-cluster` provides cluster structure that `embed-drift` can use to detect cluster-level drift:
- New cluster emergence (documents no longer fit existing clusters).
- Cluster merging (two previously distinct clusters begin overlapping).
- Centroid drift (a cluster's centroid shifts across runs, indicating content evolution).

`embed-cluster` exports cluster centroids and per-cluster quality scores in a format that `embed-drift` can consume as reference distribution snapshots.

### `memory-dedup` Integration

`memory-dedup` deduplicates semantically similar items. `embed-cluster` and `memory-dedup` are complementary: cluster first to find groups, then deduplicate within each cluster. This is more efficient than deduplicating the full corpus (deduplication is O(n^2) per cluster, not O(N^2) for the full corpus).

```typescript
import { cluster } from 'embed-cluster';
import { dedup } from 'memory-dedup';

const clusterResult = await cluster(items);
const dedupedClusters = await Promise.all(
  clusterResult.clusters.map(c => dedup(c.items, { threshold: 0.95 }))
);
```

### `fusion-rank` Integration

`fusion-rank` performs reciprocal rank fusion across multiple retrievers. `embed-cluster` cluster centroids can be used as query vectors in multi-centroid retrieval: search the vector database with each cluster centroid, fuse the results with `fusion-rank`, and rank the fused results. This "cluster-then-retrieve" approach is effective for broad queries that span multiple topics.

### `context-packer` Integration

`context-packer`'s coverage strategy internally performs k-means clustering over candidate chunks to ensure topically diverse context packing. `embed-cluster` provides the underlying clustering implementation for this strategy, ensuring consistent algorithm behavior across the monorepo.

---

## 15. Testing Strategy

### Unit Tests

Each algorithm component has isolated unit tests:

- **k-means++**: Verify that initialization produces k distinct centroids, that the second centroid is not identical to the first, and that the D^2-weighting produces points that are spread across the space (measured by average pairwise distance among initial centroids vs. random initialization).

- **Assignment step**: Verify that every point is assigned to its nearest centroid using cosine distance. Test with a trivial two-cluster case where the correct assignment is analytically known.

- **Update step**: Verify that the centroid is updated to the arithmetic mean of cluster members. For cosine clustering, verify that centroids are re-normalized. Verify that empty-cluster reinitialization fires correctly when a cluster loses all members.

- **Convergence detection**: Verify that the algorithm stops when assignments do not change. Verify that it stops at `maxIterations` even if not converged. Verify that the centroid movement threshold stops iteration correctly.

- **TF-IDF labeler**: Verify that terms present only in one cluster receive high IDF. Verify that terms present in all clusters receive IDF near 1 (minimal). Verify that stop words are excluded. Verify that bigrams are extracted correctly.

- **Silhouette score**: Test against analytically computed silhouette coefficients for a 3-point, 2-cluster case where a(x) and b(x) can be computed by hand.

- **Kneedle algorithm**: Test with a known elbow at k=4 (decreasing inertia: 100, 60, 40, 20, 18, 17, 16) -- verify elbow is detected at k=4.

- **Consensus k-selection**: Test the agree / disagree-by-1 / disagree-by-2+ cases. Verify confidence levels are assigned correctly.

### Integration Tests

- **End-to-end clustering**: Generate synthetic embeddings with known cluster structure (2D Gaussian blobs projected to high dimensions using a random orthogonal matrix). Verify that `cluster()` recovers the correct k and that cluster assignments match the ground truth at >= 95% accuracy (accounting for random initialization variance).

- **Auto-k detection accuracy**: Use the same synthetic blobs with k_true = 5. Verify that auto-k selects k = 5 (or 4 or 6 with a toleranceof ±1) for a range of n values (100, 500, 2000).

- **TF-IDF labeling integration**: Cluster a small document set with known topics. Verify that the top keyword for each cluster is a term strongly associated with the intended topic.

- **Reproducibility**: With `seed` set, verify that two calls with identical inputs produce bit-for-bit identical results.

- **EmbedItem[] passthrough**: Verify that `id` and `metadata` fields are correctly passed through to `ClusterItem` outputs.

- **Outlier detection**: Inject an obviously out-of-distribution embedding (e.g., a random vector in the opposite direction from all other vectors). Verify it appears in `ClusterResult.outliers`.

### Performance Tests

- **Speed benchmark**: Cluster 5000 items with 1536 dimensions, k = 10, nInit = 10. Target: complete in under 30 seconds on a 2021 M1 MacBook Pro. (Reference: 5000 items * 10 clusters * 1536 dims * 300 iterations * 10 restarts ≈ 23 billion operations; typed arrays in V8 run at ~5 GFLOPS, so ~4.6 seconds per run, 46 seconds for 10 restarts. PCA to 50 dims makes this 30x faster: ~1.5 seconds.)

- **Auto-k sweep performance**: Run the full auto-k sweep (k = 2 to 20) on 2000 items with 1536 dimensions. Target: complete in under 60 seconds.

- **Memory usage**: 5000 items with 1536 float64 dimensions = ~60 MB for the input matrix. With normalized copy + centroid matrices + distance matrices: target peak memory under 500 MB.

### Test Data

Tests use three categories of data:

1. **Synthetic blobs**: 2D Gaussian clusters (known ground truth) projected to high dimensions via random orthogonal projection. Controlled, verifiable.

2. **Reduced-dimensionality embeddings**: A precomputed set of 200 embeddings of real texts (from the package's test fixtures, using pre-computed vectors to avoid API calls in tests) covering 5 known topics. Used for labeling and quality metric tests.

3. **Edge cases**: Empty clusters (forced by injecting duplicate points), single-member clusters, n = kMin (minimum valid input), all-identical embeddings (degenerate input), zero vectors.

---

## 16. Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| k-means++ initialization | O(n * k) | Per restart |
| k-means iteration | O(n * k * d) | Per iteration, per restart |
| Full k-means (nInit restarts, T iterations) | O(n * k * d * T * nInit) | |
| Auto-k sweep (kMin to kMax) | O((kMax - kMin) * n * k_avg * d * T * nInit) | |
| Exact silhouette | O(n^2 * d) | Per k candidate |
| Approximate silhouette | O(n * sampleSize * d) | Per k candidate |
| TF-IDF labeling | O(n * L) | L = avg text length in tokens |
| PCA preprocessing | O(n * d * d') | d' = target dimensions |

### Memory

| Dataset | Peak Memory |
|---------|------------|
| 1000 items × 1536 dims | ~24 MB |
| 5000 items × 1536 dims | ~120 MB |
| 10000 items × 1536 dims | ~240 MB |
| 50000 items × 1536 dims | ~1.2 GB |

For datasets exceeding ~10,000 items at 1536 dimensions, PCA to 50 dimensions is strongly recommended to reduce memory and computation time. At d' = 50, 50,000 items require only ~40 MB.

### Throughput Estimates

On a 2024 M3 MacBook Pro (Apple Silicon, V8 typed array performance):

| Dataset | k | nInit | Time (no PCA) | Time (PCA d'=50) |
|---------|---|-------|---------------|-----------------|
| 500 items × 1536 dims | auto (2-16) | 10 | ~3s | ~0.5s |
| 2000 items × 1536 dims | auto (2-25) | 10 | ~20s | ~2s |
| 5000 items × 1536 dims | 10 | 10 | ~12s | ~1.5s |
| 10000 items × 1536 dims | 15 | 10 | ~45s | ~4s |

Auto-k detection multiplies single-k times by `(kMax - kMin)` for the elbow phase and adds silhouette computation on top. For large datasets, setting `kMin` and `kMax` to a narrow range or specifying k exactly eliminates this overhead.

---

## 17. Dependencies

### Runtime Dependencies

**Zero mandatory runtime dependencies.** All core algorithms (k-means, k-means++, silhouette, TF-IDF, kneedle, PCA) are implemented in pure TypeScript/JavaScript using typed arrays.

### Optional Peer Dependencies

| Package | Version | Purpose | When Needed |
|---------|---------|---------|-------------|
| `ml-pca` | `>=4.0.0` | SVD-based PCA, faster than built-in power iteration | Large datasets with many PCA components (d' > 100) or n > 100,000 |
| `umap-js` | `>=1.4.0` | UMAP dimensionality reduction for visualization | Alternative to PCA for visualization (`visualizationMethod: 'umap'`), future feature |

Peer dependencies are never auto-installed. They are detected at runtime: if `ml-pca` is available in the project's `node_modules`, it is used; otherwise the built-in PCA is used. No error is thrown if they are absent.

### DevDependencies

| Package | Purpose |
|---------|---------|
| `typescript` | TypeScript compiler |
| `vitest` | Test runner |
| `eslint` | Linting |

---

## 18. File Structure

```
embed-cluster/
├── package.json
├── tsconfig.json
├── SPEC.md
├── README.md
├── src/
│   ├── index.ts                  # Public API exports
│   ├── cluster.ts                # Main cluster() and createClusterer() functions
│   ├── kmeans/
│   │   ├── kmeans.ts             # Lloyd's algorithm implementation
│   │   ├── initialization.ts     # k-means++ and Forgy initialization
│   │   ├── distance.ts           # Cosine and Euclidean distance implementations
│   │   └── convergence.ts        # Convergence detection logic
│   ├── auto-k/
│   │   ├── auto-k.ts             # findOptimalK() coordination
│   │   ├── elbow.ts              # Elbow method + kneedle algorithm
│   │   ├── silhouette.ts         # Silhouette coefficient computation
│   │   └── calinski-harabasz.ts  # CH index tiebreaker
│   ├── labeling/
│   │   ├── tfidf.ts              # TF-IDF keyword extraction
│   │   ├── stop-words.ts         # Built-in English stop word list
│   │   └── representative.ts     # Centroid-nearest representative finder
│   ├── pca/
│   │   ├── pca.ts                # Power iteration PCA implementation
│   │   └── ml-pca-adapter.ts     # Adapter for ml-pca peer dependency
│   ├── quality/
│   │   ├── quality.ts            # ClusterQuality computation
│   │   └── outliers.ts           # Outlier detection
│   ├── visualization/
│   │   └── visualization.ts      # 2D PCA projection export
│   ├── types.ts                  # All TypeScript type definitions
│   ├── errors.ts                 # ClusterError class
│   ├── normalize.ts              # L2 normalization utilities
│   └── cli.ts                    # CLI entry point
├── test/
│   ├── kmeans.test.ts            # k-means algorithm unit tests
│   ├── initialization.test.ts    # k-means++ initialization tests
│   ├── silhouette.test.ts        # Silhouette coefficient tests
│   ├── elbow.test.ts             # Elbow method + kneedle tests
│   ├── tfidf.test.ts             # TF-IDF labeler tests
│   ├── auto-k.test.ts            # Auto-k consensus tests
│   ├── cluster.test.ts           # End-to-end cluster() integration tests
│   ├── pca.test.ts               # PCA implementation tests
│   ├── visualization.test.ts     # Visualization export tests
│   ├── cli.test.ts               # CLI integration tests
│   └── fixtures/
│       ├── embeddings-5topics.json   # Precomputed embeddings for 5 known topics
│       ├── embeddings-random.json    # Random embeddings (no cluster structure)
│       └── embeddings-small.json    # Small set for fast tests
└── dist/                         # Build output (gitignored)
    ├── index.js
    ├── index.d.ts
    └── ...
```

---

## 19. Implementation Roadmap

### Phase 1: Core Clustering (v0.1.0)

- Implement k-means with k-means++ initialization, cosine and Euclidean distance, convergence detection, and multiple restarts.
- Implement `cluster(embeddings, { k })` with fixed k.
- Compute per-point silhouette coefficients and cluster quality metrics.
- Return `ClusterResult` with clusters, centroids, inertia, and silhouette scores.
- Full unit tests for k-means and silhouette.

### Phase 2: Auto-k Detection (v0.2.0)

- Implement the elbow method with kneedle algorithm.
- Implement silhouette-method auto-k.
- Implement CH-index tiebreaker.
- Implement `findOptimalK()` as a standalone function.
- Wire auto-k into `cluster()` as the default when k is not specified.
- Integration tests with synthetic Gaussian blobs.

### Phase 3: Topic Labeling (v0.3.0)

- Implement TF-IDF keyword extraction with stop word filtering, bigram support, and `labelConfidence` scoring.
- Implement centroid-nearest representative selection.
- Add the `labeler` custom function hook.
- `cluster(items: EmbedItem[])` overload with labeling pipeline.
- Integration tests with precomputed embedding fixtures.

### Phase 4: PCA and Visualization (v0.4.0)

- Implement built-in power-iteration PCA.
- Add `ml-pca` peer dependency adapter for large datasets.
- Implement 2D PCA projection for visualization export.
- `computeVisualization` option and `VisualizationData` output.
- Performance benchmarks; verify targets from Section 16.

### Phase 5: CLI (v0.5.0)

- Implement `embed-cluster` CLI binary.
- Support `--input`, `--output`, `--format`, `--k`, `--k-min`, `--k-max`, `--metric`, `--pca`, `--viz`.
- Human-readable summary output format.
- CLI integration tests.

### Phase 6: Parallel Restarts and Production Hardening (v0.6.0)

- Implement `parallel: true` mode using Node.js worker threads for parallel k-means restarts.
- Approximate silhouette with configurable `silhouetteSampleSize`.
- Outlier detection and `silhouetteOutlierThreshold` option.
- Empty-cluster reinitalization robustness.
- `createClusterer()` factory.
- Full performance test suite; optimize hot paths.

### Phase 7: Monorepo Integration (v1.0.0)

- `embed-cache` integration tests (cluster embeddings produced by embed-cache).
- `embed-drift` centroid snapshot export format.
- `memory-dedup` pipeline integration test.
- `context-packer` coverage strategy: extract clustering core and integrate.
- API stability review and documentation pass.
- Publish to npm.

---

## 20. Example Use Cases (Detailed Walkthroughs)

### Document Topic Discovery in a RAG Pipeline

**Problem**: An engineering team has indexed 8,000 internal documents into a vector database. Before exposing search, they want to understand the topic distribution: how many distinct engineering topics exist, and what are they?

**Setup**:
```typescript
import { createCache } from 'embed-cache';
import { cluster } from 'embed-cluster';

const cache = createCache({ embedder: openaiEmbed, model: 'text-embedding-3-small' });

// Batch embed all documents (mostly cache hits after the first run)
const texts = documents.map(d => d.summary ?? d.title);
const embeddings = await cache.embedBatch(texts);

const items = documents.map((doc, i) => ({
  id: doc.id,
  text: texts[i],
  embedding: embeddings[i],
  metadata: { url: doc.url, team: doc.team },
}));
```

**Clustering**:
```typescript
const result = await cluster(items, {
  kMin: 5,
  kMax: 40,
  pca: { enabled: true, dimensions: 50 },
  computeVisualization: true,
});
```

**Interpreting results**:
```typescript
console.log(`Discovered ${result.k} topics (${result.quality.interpretation} quality)`);
console.log(`Auto-k confidence: ${result.autoK?.confidence}`);

for (const c of result.clusters.sort((a, b) => b.size - a.size)) {
  const keywords = c.keywords?.slice(0, 3).map(k => k.term).join(', ');
  console.log(`  [${c.size} docs] ${c.label} (quality: ${c.silhouetteScore.toFixed(2)})`);
  console.log(`    Keywords: ${keywords}`);
  console.log(`    Example: ${c.representative?.text?.slice(0, 80)}`);
}

// Export visualization for the engineering wiki
fs.writeFileSync('topic-map.json', JSON.stringify(result.visualization));
```

**Typical output** for an engineering document corpus:
```
Discovered 14 topics (good quality)
Auto-k confidence: high

  [812 docs] kubernetes, deployment, helm (quality: 0.71)
    Keywords: kubernetes, helm, deployment, pod, cluster
    Example: Deploying microservices to Kubernetes using Helm charts...

  [734 docs] python, testing, pytest (quality: 0.68)
    Keywords: python, pytest, testing, mock, fixtures
    Example: Setting up pytest with fixtures and mock objects...
  ...
```

### Content Taxonomy for a Publishing Platform

**Problem**: A publishing platform has 50,000 articles and needs a content taxonomy for navigation. Manual taxonomy construction is infeasible.

**Strategy**: Cluster at two levels -- first find 20-30 top-level topics, then sub-cluster each topic to find 3-10 subtopics each.

```typescript
import { cluster } from 'embed-cluster';

// Level 1: top-level taxonomy
const topLevel = await cluster(articleItems, {
  kMin: 15, kMax: 35,
  pca: { enabled: true, dimensions: 50 },
});

// Level 2: sub-topics within each top-level cluster
const taxonomy = await Promise.all(
  topLevel.clusters.map(async (c) => {
    if (c.size < 50) return { ...c, subtopics: [] }; // too small to sub-cluster

    const subResult = await cluster(c.items.map(i => ({
      id: i.id,
      text: i.text!,
      embedding: i.embedding,
    })), {
      kMin: 2,
      kMax: Math.min(10, Math.floor(c.size / 20)),
    });

    return { ...c, subtopics: subResult.clusters };
  })
);
```

### Anomaly Detection in User Submissions

**Problem**: A moderation system needs to flag unusual submissions without labeled training data.

```typescript
import { cluster } from 'embed-cluster';

const result = await cluster(submissions.map(s => ({
  id: s.id,
  text: s.content,
  embedding: s.embedding,
})), {
  k: 8, // known number of normal submission categories
  silhouetteOutlierThreshold: -0.1, // more aggressive than default 0
});

// Outliers: submissions that don't fit any normal category
const flagged = result.outliers.map(o => ({
  id: o.id,
  text: o.text,
  silhouette: o.silhouette,
  bestCluster: result.clusters[o.clusterId].label,
}));

console.log(`Flagged ${flagged.length} submissions for review`);
```

### Periodic Re-Clustering for Drift Detection

**Problem**: A memory system wants to detect when new topic clusters emerge in the agent's observations over time.

```typescript
import { cluster } from 'embed-cluster';

async function recluster(memories: EmbedItem[], previousResult?: ClusterResult) {
  const current = await cluster(memories, {
    seed: 42, // reproducible for comparison
    computeVisualization: false,
  });

  if (previousResult) {
    // Detect new clusters: centroids in current that are far from all previous centroids
    const threshold = 0.4;
    for (const currentCluster of current.clusters) {
      const minDist = Math.min(
        ...previousResult.clusters.map(prev =>
          cosineDistance(currentCluster.centroid, prev.centroid)
        )
      );
      if (minDist > threshold) {
        console.log(`New topic emerged: ${currentCluster.label}`);
      }
    }
  }

  return current;
}
```

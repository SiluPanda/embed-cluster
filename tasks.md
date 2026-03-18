# embed-cluster — Task Breakdown

This file tracks all implementation tasks derived from SPEC.md. Each task is granular, actionable, and grouped by phase.

---

## Phase 1: Project Setup and Scaffolding

- [ ] **Install dev dependencies** — Add `typescript`, `vitest`, and `eslint` to devDependencies in package.json. Install them. | Status: not_done
- [ ] **Configure ESLint** — Create an ESLint configuration file suitable for TypeScript. Follow monorepo conventions if one exists. | Status: not_done
- [ ] **Add vitest config** — Create a `vitest.config.ts` (or configure vitest in package.json) to handle the test/ directory and TypeScript source. | Status: not_done
- [ ] **Create directory structure** — Create all directories specified in the spec: `src/kmeans/`, `src/auto-k/`, `src/labeling/`, `src/pca/`, `src/quality/`, `src/visualization/`, and `test/fixtures/`. | Status: not_done
- [ ] **Create types.ts** — Define all TypeScript interfaces and types: `EmbedItem`, `ClusterOptions`, `ClusterResult`, `Cluster`, `ClusterItem`, `SilhouetteResult`, `OptimalKResult`, `ClusterQuality`, `VisualizationData`, `LabelerFn`, `Clusterer`. Exactly as specified in Section 9 of the spec. | Status: not_done
- [ ] **Create errors.ts** — Implement `ClusterError` class extending `Error` with `code` field supporting values: `EMPTY_INPUT`, `INCONSISTENT_DIMENSIONS`, `DEGENERATE_INPUT`, `INVALID_K`, `INVALID_OPTIONS`. | Status: not_done
- [ ] **Create normalize.ts** — Implement L2 normalization utility: takes a `number[]` vector and returns a unit-length copy. Handle the zero-vector degenerate case (return zero vector unchanged). Also implement a batch normalization function for `number[][]`. | Status: not_done
- [ ] **Set up index.ts exports** — Replace the placeholder `export {}` with proper re-exports of the public API surface: `cluster`, `findOptimalK`, `silhouetteScore`, `createClusterer`, all types, and `ClusterError`. (Exports will be wired up as implementations land.) | Status: not_done
- [ ] **Add bin entry to package.json** — Add a `"bin"` field pointing to the CLI entry point (`dist/cli.js`) for the `embed-cluster` command. | Status: not_done
- [ ] **Add optional peerDependencies to package.json** — Add `ml-pca` (>=4.0.0) and `umap-js` (>=1.4.0) as optional peer dependencies with `peerDependenciesMeta` marking them as optional. | Status: not_done
- [ ] **Create test fixture: embeddings-small.json** — Generate a small set of synthetic embedding vectors (e.g., 20 vectors in 8 dimensions with 3 clear clusters) for fast unit tests. | Status: not_done
- [ ] **Create test fixture: embeddings-5topics.json** — Generate or precompute 200 embedding vectors covering 5 known topics with associated text, for integration and labeling tests. | Status: not_done
- [ ] **Create test fixture: embeddings-random.json** — Generate random embedding vectors with no cluster structure, for edge-case and quality metric tests. | Status: not_done

---

## Phase 2: Distance Functions (src/kmeans/distance.ts)

- [ ] **Implement cosine similarity** — `cosineSimilarity(u: number[], v: number[]): number` computing `dot(u,v) / (||u|| * ||v||)`. | Status: not_done
- [ ] **Implement cosine distance** — `cosineDistance(u: number[], v: number[]): number` returning `1 - cosineSimilarity(u, v)`. For pre-normalized vectors, this simplifies to `1 - dot(u, v)`. | Status: not_done
- [ ] **Implement Euclidean distance** — `euclideanDistance(u: number[], v: number[]): number` computing `sqrt(sum((u[i] - v[i])^2))`. | Status: not_done
- [ ] **Implement squared Euclidean distance** — `euclideanDistanceSquared(u: number[], v: number[]): number` (avoids the sqrt for performance in assignment steps). | Status: not_done
- [ ] **Implement dot product** — `dotProduct(u: number[], v: number[]): number` as a standalone utility used by cosine computations and the vectorized assignment step. | Status: not_done
- [ ] **Implement distance factory** — `getDistanceFn(metric: 'cosine' | 'euclidean')` returning the appropriate distance function. | Status: not_done
- [ ] **Write unit tests for distance functions** — Test cosine and Euclidean distance with known values, including edge cases (zero vectors, identical vectors, opposite vectors, unit vectors). | Status: not_done

---

## Phase 3: k-means++ Initialization (src/kmeans/initialization.ts)

- [ ] **Implement k-means++ initialization** — Given data points and k, select k initial centroids using the D^2-weighted probabilistic selection described in Section 4. First centroid chosen uniformly at random; subsequent centroids chosen with probability proportional to squared distance to nearest existing centroid. | Status: not_done
- [ ] **Implement Forgy (random) initialization** — Select k data points uniformly at random as initial centroids. Available as an alternative for reproducibility testing. | Status: not_done
- [ ] **Implement seeded random number generator** — Create a deterministic PRNG that accepts an optional seed for reproducible initialization. When no seed is provided, use `Math.random()`. | Status: not_done
- [ ] **Write unit tests for k-means++ initialization** — Verify: produces exactly k distinct centroids; second centroid is not identical to first; D^2-weighting produces spread-out centroids (measure average pairwise distance vs. random init); seeded init is reproducible. | Status: not_done
- [ ] **Write unit tests for Forgy initialization** — Verify: produces k centroids; each centroid is a data point; seeded init is reproducible. | Status: not_done

---

## Phase 4: Core k-means Algorithm (src/kmeans/kmeans.ts, src/kmeans/convergence.ts)

- [ ] **Implement assignment step** — For each point, compute distance to all centroids and assign to the nearest centroid. For cosine distance on normalized vectors, use maximum dot product. Return an array of cluster labels (0-indexed). | Status: not_done
- [ ] **Implement vectorized assignment step** — Implement the assignment step as a batch matrix-dot-product computation (`X @ C.T`) for efficiency with typed arrays, then argmax each row. | Status: not_done
- [ ] **Implement centroid update step** — Recompute each centroid as the arithmetic mean of its assigned points. For cosine distance mode, re-normalize centroids to unit length after averaging. | Status: not_done
- [ ] **Implement empty cluster reinitialization** — When a cluster loses all members during update, reinitialize its centroid by selecting the data point with the highest inertia contribution (farthest from its assigned centroid). | Status: not_done
- [ ] **Implement convergence detection** — Detect convergence via three criteria: (1) no assignment changes, (2) maximum centroid movement below `tolerance` (default 1e-4), (3) iteration count exceeds `maxIterations` (default 300). | Status: not_done
- [ ] **Implement single k-means run** — Combine initialization, assignment, update, and convergence into a single `kmeansRun(data, k, options)` function. Return labels, centroids, inertia, and iteration count. | Status: not_done
- [ ] **Implement inertia computation** — After convergence, compute total inertia as the within-cluster sum of squared distances from each point to its centroid. | Status: not_done
- [ ] **Implement multiple restarts** — Run `nInit` (default 10) independent k-means runs with different random seeds. Keep the run with the lowest inertia. | Status: not_done
- [ ] **Support cosine mode pre-normalization** — When `distanceMetric: 'cosine'`, L2-normalize all input vectors once before the first run. Reuse the normalized copy across all restarts. Do not mutate the original input. | Status: not_done
- [ ] **Write unit tests: assignment step** — Test with a trivial 2-cluster case where correct assignment is analytically known. Verify every point is assigned to its nearest centroid. | Status: not_done
- [ ] **Write unit tests: centroid update** — Verify centroid is the arithmetic mean of cluster members. For cosine mode, verify re-normalization. Verify empty cluster reinitialization fires correctly. | Status: not_done
- [ ] **Write unit tests: convergence detection** — Verify algorithm stops when assignments do not change. Verify it stops at maxIterations. Verify centroid movement threshold works. | Status: not_done
- [ ] **Write unit tests: multiple restarts** — Verify the run with lowest inertia is selected. Verify nInit restarts are actually executed. | Status: not_done
- [ ] **Write unit tests: inertia computation** — Verify inertia against hand-computed values for a small dataset. | Status: not_done

---

## Phase 5: Silhouette Score Computation (src/auto-k/silhouette.ts)

- [ ] **Implement per-point silhouette coefficient** — For each point x in cluster C_i: compute a(x) = mean distance to all other points in C_i, b(x) = min over other clusters C_j of mean distance to all points in C_j, s(x) = (b(x) - a(x)) / max(a(x), b(x)). Handle single-member clusters (s(x) = 0). | Status: not_done
- [ ] **Implement exact silhouette computation** — Compute silhouette for all points using full pairwise distances. Used when n <= 1000. | Status: not_done
- [ ] **Implement approximate silhouette via reservoir sampling** — For n > 1000, compute a(x) and b(x) using a random sample of `silhouetteSampleSize` (default 500) points per cluster instead of all points. | Status: not_done
- [ ] **Implement per-cluster silhouette score** — Mean silhouette coefficient across all points in each cluster. Return as `Record<number, number>`. | Status: not_done
- [ ] **Implement mean silhouette score** — Mean silhouette coefficient across all points. | Status: not_done
- [ ] **Implement public `silhouetteScore()` function** — Accepts `embeddings: number[][]` and `labels: number[]` with optional distance metric and sample size. Returns `SilhouetteResult` with `meanScore`, `scores`, `clusterScores`, and `outlierIndices`. | Status: not_done
- [ ] **Write unit tests for silhouette** — Test against analytically computed values for a 3-point, 2-cluster case. Test single-member cluster edge case. Test approximate vs. exact silhouette agreement within tolerance. | Status: not_done

---

## Phase 6: Elbow Method and Kneedle Algorithm (src/auto-k/elbow.ts)

- [ ] **Implement inertia sweep** — For each k from kMin to kMax, run full k-means with nInit restarts and collect resulting inertia values as `Record<number, number>`. | Status: not_done
- [ ] **Implement kneedle algorithm** — Given the (k, inertia) curve: (1) normalize both axes to [0, 1], (2) compute difference from the diagonal connecting the curve endpoints, (3) the elbow is the k that maximizes this difference. | Status: not_done
- [ ] **Return elbowK** — The k recommended by the elbow/kneedle method. | Status: not_done
- [ ] **Write unit tests for kneedle** — Test with a known elbow at k=4 using the inertia sequence [100, 60, 40, 20, 18, 17, 16]. Verify elbow is detected at k=4. Test smooth curves with no sharp elbow. | Status: not_done

---

## Phase 7: Calinski-Harabasz Index (src/auto-k/calinski-harabasz.ts)

- [ ] **Implement Calinski-Harabasz index** — Compute CH(k) = [SSB / (k-1)] / [SSW / (n-k)] where SSB is between-cluster sum of squares and SSW is within-cluster sum of squares (inertia). | Status: not_done
- [ ] **Implement SSB computation** — Sum of squared distances from each cluster centroid to the global centroid, weighted by cluster size. | Status: not_done
- [ ] **Write unit tests for CH index** — Test with known cluster structures where the CH index should prefer the correct k. | Status: not_done

---

## Phase 8: Auto-K Consensus and findOptimalK (src/auto-k/auto-k.ts)

- [ ] **Implement silhouette method k-selection** — Run k-means for each k in [kMin, kMax], compute mean silhouette for each k, select the k with the highest mean silhouette score. | Status: not_done
- [ ] **Implement consensus decision logic** — Combine elbow and silhouette recommendations: (1) both agree -> use that k, confidence: high; (2) disagree by 1 -> prefer silhouette k, confidence: medium; (3) disagree by 2+ -> compute CH index for top-2 candidates, pick higher CH, confidence: low. | Status: not_done
- [ ] **Implement `findOptimalK()` function** — Public function that runs only the k-detection phase. Accepts embeddings and options (kMin, kMax, nInit, distanceMetric, silhouetteSampleSize, methods). Returns `OptimalKResult` with elected k, confidence, elbowK, silhouetteK, tiebreakerK, elbowInertias, silhouetteScores. | Status: not_done
- [ ] **Support `autoKMethods` option** — Allow callers to run only elbow, only silhouette, or both methods. When only one method is specified, its recommendation is used directly with confidence: medium. | Status: not_done
- [ ] **Implement default kMax computation** — `kMax = min(floor(sqrt(n)), 30)` when not specified by the caller. | Status: not_done
- [ ] **Write unit tests for consensus logic** — Test agree, disagree-by-1, and disagree-by-2+ scenarios. Verify correct confidence levels. Verify tiebreaker invocation. | Status: not_done
- [ ] **Write integration test for auto-k with synthetic blobs** — Generate Gaussian blobs with known k_true = 5. Verify auto-k selects k within +/-1 of ground truth for n = 100, 500, 2000. | Status: not_done

---

## Phase 9: Input Validation and Error Handling

- [ ] **Validate empty input** — Throw `ClusterError` with code `EMPTY_INPUT` if the input array is empty or contains fewer than `kMin + 1` points. | Status: not_done
- [ ] **Validate consistent dimensions** — Throw `ClusterError` with code `INCONSISTENT_DIMENSIONS` if embedding vectors have different lengths. | Status: not_done
- [ ] **Validate degenerate input** — Throw `ClusterError` with code `DEGENERATE_INPUT` if all embedding vectors are zero vectors. | Status: not_done
- [ ] **Validate k value** — Throw `ClusterError` with code `INVALID_K` if user-specified k < 2 or k > number of input points. | Status: not_done
- [ ] **Validate kMin/kMax** — Throw `ClusterError` with code `INVALID_OPTIONS` if `kMin >= kMax` when both are specified during auto-detection. | Status: not_done
- [ ] **Validate ClusterOptions fields** — Validate nInit > 0, maxIterations > 0, tolerance > 0, silhouetteSampleSize > 0, pca.dimensions > 0. Throw `ClusterError` with code `INVALID_OPTIONS` for invalid values. | Status: not_done
- [ ] **Write unit tests for all validation cases** — Test each error condition individually. Verify correct error code and message. | Status: not_done

---

## Phase 10: Main cluster() Function (src/cluster.ts)

- [ ] **Implement `cluster()` for `number[][]` input** — Accept raw embeddings, validate input, optionally run PCA preprocessing, auto-detect or use provided k, run k-means with restarts, compute silhouette scores, compute quality metrics, compute inter-cluster distances, detect outliers, and return `ClusterResult`. | Status: not_done
- [ ] **Implement `cluster()` for `EmbedItem[]` input** — Detect that input contains `EmbedItem` objects (by checking for `embedding` property). Extract embeddings for clustering, then attach text, id, and metadata back to `ClusterItem` results. Run TF-IDF labeling pipeline after clustering. | Status: not_done
- [ ] **Implement input type detection** — Distinguish between `number[][]` and `EmbedItem[]` input. If the first element has an `embedding` property, treat as `EmbedItem[]`. | Status: not_done
- [ ] **Wire auto-k into cluster()** — When `k` is not specified, call `findOptimalK()` to determine k, then cluster at that k. Attach `autoK` result to `ClusterResult`. | Status: not_done
- [ ] **Wire user-specified k** — When `k` is specified, skip auto-detection and cluster directly at that k. `autoK` is undefined in the result. | Status: not_done
- [ ] **Build ClusterResult** — Assemble the full result object: k, clusters array, quality, inertia, silhouetteScore, interClusterDistances, outliers, autoK, visualization. | Status: not_done
- [ ] **Build Cluster objects** — For each cluster: id, centroid, items, size, inertia, silhouetteScore. When text is available: label, keywords, labelConfidence, representative. | Status: not_done
- [ ] **Build ClusterItem objects** — For each point: clusterId, id (from EmbedItem.id or sequential index), embedding (original, before normalization), text, metadata, silhouette, distanceToCentroid, projection. | Status: not_done
- [ ] **Compute inter-cluster distances** — Build the k x k symmetric matrix of pairwise cosine distances between centroids. | Status: not_done
- [ ] **Write end-to-end integration test** — Cluster synthetic Gaussian blobs, verify correct k recovery, >= 95% assignment accuracy. | Status: not_done
- [ ] **Write integration test for EmbedItem[] passthrough** — Verify id, text, and metadata fields are correctly preserved in ClusterItem output. | Status: not_done
- [ ] **Write reproducibility test** — With `seed` set, verify two identical calls produce bit-for-bit identical results. | Status: not_done

---

## Phase 11: Quality Metrics (src/quality/quality.ts, src/quality/outliers.ts)

- [ ] **Implement ClusterQuality computation** — Compute all fields: meanSilhouette, minClusterSilhouette, maxClusterSilhouette, totalInertia, meanInterClusterDistance, minInterClusterDistance, outlierRate, sizeCV, interpretation. | Status: not_done
- [ ] **Implement quality interpretation rubric** — excellent: meanSilhouette >= 0.7 and outlierRate <= 0.05; good: meanSilhouette >= 0.5 and outlierRate <= 0.15; fair: meanSilhouette >= 0.25 and outlierRate <= 0.30; poor: otherwise. | Status: not_done
- [ ] **Implement cluster size CV** — Compute coefficient of variation (std dev / mean) of cluster sizes. | Status: not_done
- [ ] **Implement outlier detection** — Identify all points with silhouette < `silhouetteOutlierThreshold` (default 0). Collect into `ClusterResult.outliers`. Compute `outlierRate` as fraction of total points. | Status: not_done
- [ ] **Write unit tests for quality metrics** — Test interpretation rubric with known silhouette/outlier rate combinations. Test sizeCV computation. Test outlier detection with injected outlier points. | Status: not_done
- [ ] **Write integration test for outlier detection** — Inject an obviously out-of-distribution embedding (random vector opposite all others). Verify it appears in `ClusterResult.outliers`. | Status: not_done

---

## Phase 12: TF-IDF Topic Labeling (src/labeling/tfidf.ts, src/labeling/stop-words.ts, src/labeling/representative.ts)

- [ ] **Implement tokenizer** — Lowercase text, strip punctuation, split on whitespace. Discard tokens shorter than 3 characters. | Status: not_done
- [ ] **Implement stop word list** — Create a built-in list of ~200 common English function words (the, is, a, of, and, to, in, that, for, it, are, this, was, be, as, etc.). Support `additionalStopWords` option for extension. | Status: not_done
- [ ] **Implement stop word filtering** — Filter out stop words from token lists before TF-IDF scoring. | Status: not_done
- [ ] **Implement TF computation** — `TF(term, cluster) = count(term in cluster) / total_tokens_in_cluster`. Treat each cluster's combined texts as a single "document." | Status: not_done
- [ ] **Implement IDF computation** — `IDF(term) = log((1 + k) / (1 + count(clusters containing term))) + 1`. The +1 smoothing handles terms in all clusters and the single-cluster edge case. | Status: not_done
- [ ] **Implement TF-IDF scoring** — `score(term, cluster) = TF(term, cluster) * IDF(term)`. Rank terms by score per cluster. | Status: not_done
- [ ] **Implement bigram extraction** — Extract bigrams from adjacent non-stop-word tokens. Score bigrams using the same TF-IDF formula. Prefer bigrams over unigrams when their score exceeds a threshold. | Status: not_done
- [ ] **Implement top-N keyword extraction** — Return the top `labelTopN` (default 5) terms/bigrams by TF-IDF score as `Array<{ term: string; score: number }>`. | Status: not_done
- [ ] **Implement human-readable label string** — Join the top 3 terms with ", " to form the cluster's `label` string. | Status: not_done
- [ ] **Implement label confidence scoring** — Compute `labelConfidence` per cluster as the normalized TF-IDF score of the top keyword relative to the maximum TF-IDF score across all clusters. Range [0, 1]. | Status: not_done
- [ ] **Implement centroid-nearest representative** — For each cluster, find the item whose embedding is closest to the centroid (argmin cosine distance). Store as `Cluster.representative`. | Status: not_done
- [ ] **Implement custom labeler hook** — When `labeling.labeler` is provided, call it for each cluster with `{ items, centroid, representative, id }`. Call concurrently for all clusters via `Promise.all`. The returned string becomes `Cluster.label`. Skip TF-IDF when custom labeler is used. | Status: not_done
- [ ] **Write unit tests for tokenizer** — Test with various inputs: punctuation, mixed case, short tokens, whitespace variations. | Status: not_done
- [ ] **Write unit tests for TF-IDF** — Verify terms present only in one cluster receive high IDF. Verify terms in all clusters get IDF near 1. Verify stop words are excluded. Verify bigram extraction. | Status: not_done
- [ ] **Write unit tests for representative selection** — Verify the selected representative is indeed the closest item to the centroid. | Status: not_done
- [ ] **Write integration test for TF-IDF labeling** — Cluster a small document set with known topics. Verify the top keyword per cluster is a term strongly associated with the intended topic. | Status: not_done

---

## Phase 13: PCA Dimensionality Reduction (src/pca/pca.ts, src/pca/ml-pca-adapter.ts)

- [ ] **Implement data centering** — Subtract the mean vector from each data point. Store the mean for inverse projection. | Status: not_done
- [ ] **Implement power iteration for single component** — Initialize a random unit vector, repeatedly multiply by `X^T @ (X @ v)`, normalize, until convergence. Return the principal component vector. | Status: not_done
- [ ] **Implement deflation** — After extracting a component, remove its variance from the data: `X = X - (X @ v) * v^T`. | Status: not_done
- [ ] **Implement full PCA via power iteration** — Extract d' (default 50) principal components sequentially using power iteration + deflation. Return the projection matrix (d x d'). | Status: not_done
- [ ] **Implement data projection** — Project the centered data onto the d' principal components: `X' = X_centered @ V`. | Status: not_done
- [ ] **Implement inverse projection** — Project low-dimensional centroids back to the original space: `centroid_full = centroid_low @ V^T + mean`. Ensures output centroids are in the original embedding space. | Status: not_done
- [ ] **Implement variance explained computation** — Compute the fraction of total variance captured by the d' components: `sum(eigenvalues[0..d']) / sum(all eigenvalues)`. | Status: not_done
- [ ] **Implement ml-pca adapter** — Detect if `ml-pca` is available via `require.resolve`. If available, use it for SVD-based PCA. If not, fall back to built-in power iteration. No error thrown on absence. | Status: not_done
- [ ] **Wire PCA into cluster()** — When `pca.enabled: true`, project input data before clustering. Project centroids back to original space for output. | Status: not_done
- [ ] **Write unit tests for PCA** — Test on a dataset with known principal components. Verify variance explained is correct. Verify inverse projection recovers approximate original coordinates. Test with d' = 2 for visualization. | Status: not_done

---

## Phase 14: Visualization Export (src/visualization/visualization.ts)

- [ ] **Implement 2D PCA projection for visualization** — When `computeVisualization: true`, project all data points to 2D using PCA (top 2 components). Also project cluster centroids to 2D. | Status: not_done
- [ ] **Build VisualizationData object** — Construct `points` array with `{ x, y, clusterId, id, text? }` for each item. Construct `centroids` array with `{ x, y, clusterId, label? }`. Compute `varianceExplained` for the 2D projection. | Status: not_done
- [ ] **Attach projection to ClusterItem** — Set `ClusterItem.projection = { x, y }` for each item when visualization is enabled. | Status: not_done
- [ ] **Write unit tests for visualization** — Verify output structure matches `VisualizationData`. Verify varianceExplained is in [0, 1]. Verify centroid projections are positioned among their cluster members. | Status: not_done

---

## Phase 15: createClusterer Factory

- [ ] **Implement `createClusterer(config)`** — Factory function that accepts `ClusterOptions` and returns a `Clusterer` instance with pre-bound configuration. The instance exposes `cluster(embeddings)`, `cluster(items)`, and `findOptimalK(embeddings)` methods that merge the pre-configured options with any per-call overrides. | Status: not_done
- [ ] **Write unit tests for createClusterer** — Verify pre-configured options are applied. Verify per-call options can override pre-configured ones. | Status: not_done

---

## Phase 16: CLI (src/cli.ts)

- [ ] **Implement CLI argument parsing** — Parse all flags from Section 12 of the spec: `--input`, `--output`, `--format`, `--k`, `--k-min`, `--k-max`, `--metric`, `--n-init`, `--max-iterations`, `--seed`, `--methods`, `--silhouette-sample`, `--top-n`, `--no-bigrams`, `--pca`, `--pca-dims`, `--viz`, `--version`, `--help`. | Status: not_done
- [ ] **Implement --help output** — Print usage information with all flags and descriptions. | Status: not_done
- [ ] **Implement --version output** — Read version from package.json and print it. | Status: not_done
- [ ] **Implement JSON input parsing** — Read the `--input` file, parse as JSON, detect whether it is `number[][]` or `EmbedItem[]`. Handle file-not-found and invalid-JSON errors. | Status: not_done
- [ ] **Implement summary output format** — Format `ClusterResult` as the human-readable terminal output shown in Section 12 of the spec. Include input info, auto-k results, clustering results, cluster list with sizes/silhouette/labels, outlier count, and quality interpretation. | Status: not_done
- [ ] **Implement JSON output format** — Write the full `ClusterResult` as pretty-printed JSON to stdout or `--output` file. | Status: not_done
- [ ] **Implement --viz flag** — When `--viz <path>` is provided, export the `VisualizationData` to the specified path as JSON. Automatically enable `computeVisualization: true`. | Status: not_done
- [ ] **Implement exit codes** — Exit 0 on success, 1 on input error, 2 on configuration error, 3 on clustering warning (low confidence auto-k or 'poor' quality). | Status: not_done
- [ ] **Add shebang line** — Add `#!/usr/bin/env node` to the top of cli.ts so the compiled JS can be executed directly. | Status: not_done
- [ ] **Write CLI integration tests** — Test with each input format (number[][], EmbedItem[]). Test --format summary and --format json. Test --k, --k-min, --k-max. Test --pca. Test --viz. Test error exit codes for missing/invalid input. Test --help and --version. | Status: not_done

---

## Phase 17: Parallel Restarts (Optional, v0.6.0)

- [ ] **Implement worker thread pool** — Create a worker thread pool that can run k-means restart functions in parallel. Each worker receives the data, k, seed, and options, and returns labels, centroids, inertia. | Status: not_done
- [ ] **Implement parallel restart coordination** — When `parallel: true`, distribute nInit restarts across worker threads. Collect results and select the run with lowest inertia. | Status: not_done
- [ ] **Implement worker script** — Create a separate worker script (or inline Worker code) that runs a single k-means run and returns the result via `parentPort.postMessage`. | Status: not_done
- [ ] **Graceful fallback** — If worker threads fail or are unavailable, fall back to sequential restarts with a console warning. | Status: not_done
- [ ] **Write tests for parallel mode** — Verify parallel results match sequential results for the same seed. Verify speedup for large datasets. | Status: not_done

---

## Phase 18: Performance Optimization

- [ ] **Use Float64Array typed arrays for data storage** — Store the input data matrix and centroid matrix as contiguous typed arrays for cache-efficient access and V8 optimization. | Status: not_done
- [ ] **Optimize distance computation inner loop** — Ensure the dot product / distance computation uses a tight loop over typed arrays without unnecessary allocations. | Status: not_done
- [ ] **Avoid allocations in hot paths** — Pre-allocate label arrays, distance arrays, and centroid buffers. Reuse across iterations and restarts. | Status: not_done
- [ ] **Write performance benchmarks** — Benchmark: 5000 items x 1536 dims, k=10, nInit=10 (target < 30s). Auto-k sweep k=2..20 on 2000 items x 1536 dims (target < 60s). Measure peak memory for 5000 items x 1536 dims (target < 500 MB). | Status: not_done

---

## Phase 19: Edge Cases and Robustness

- [ ] **Handle n equals kMin (minimum valid input)** — Ensure clustering works when the number of points equals kMin + 1 (e.g., 3 points with kMin=2). | Status: not_done
- [ ] **Handle all-identical embeddings** — If all input vectors are identical, return a single cluster (k=1 effectively) or throw `DEGENERATE_INPUT` depending on spec interpretation. The spec says to throw; implement accordingly. | Status: not_done
- [ ] **Handle duplicate points** — Multiple identical embedding vectors should be handled gracefully without crashing or producing incorrect results. | Status: not_done
- [ ] **Handle single-member clusters in silhouette** — A cluster with 1 member has undefined intra-cluster distance. Return silhouette = 0 for that point. | Status: not_done
- [ ] **Handle k = number of points** — When the user specifies k equal to n, each point is its own cluster. Silhouette is 0 for all points. Handle gracefully. | Status: not_done
- [ ] **Handle very high dimensional input without PCA** — Ensure correctness (not just performance) for d=3072. No precision issues with Float64. | Status: not_done
- [ ] **Write edge case tests** — Cover all the above edge cases with specific test cases. | Status: not_done

---

## Phase 20: Public API Exports (src/index.ts)

- [ ] **Export `cluster` function** — The primary API. | Status: not_done
- [ ] **Export `findOptimalK` function** — Standalone k-detection. | Status: not_done
- [ ] **Export `silhouetteScore` function** — Standalone silhouette evaluation. | Status: not_done
- [ ] **Export `createClusterer` factory** — Pre-configured clusterer. | Status: not_done
- [ ] **Export all TypeScript types** — `EmbedItem`, `ClusterOptions`, `ClusterResult`, `Cluster`, `ClusterItem`, `SilhouetteResult`, `OptimalKResult`, `ClusterQuality`, `VisualizationData`, `LabelerFn`, `Clusterer`. | Status: not_done
- [ ] **Export `ClusterError`** — Error class for consumer catch blocks. | Status: not_done

---

## Phase 21: Documentation

- [ ] **Write README.md** — Include: package overview, installation, quick start, API reference for all public functions and types, configuration options table, CLI usage, integration examples (embed-cache, embed-drift, memory-dedup, fusion-rank), performance characteristics, and license. | Status: not_done
- [ ] **Add JSDoc to all public functions** — Document parameters, return types, throws clauses, and usage examples in JSDoc format for IDE intellisense. | Status: not_done
- [ ] **Add JSDoc to all public types** — Document each field in all exported interfaces with descriptions matching the spec. | Status: not_done
- [ ] **Add inline code comments** — Comment non-obvious algorithm steps (k-means++ D^2 weighting, kneedle normalization, TF-IDF IDF smoothing, etc.) for maintainability. | Status: not_done

---

## Phase 22: Build, Lint, and CI Verification

- [ ] **Verify `npm run build` succeeds** — TypeScript compilation produces dist/ output with .js, .d.ts, and .d.ts.map files. | Status: not_done
- [ ] **Verify `npm run lint` passes** — No ESLint errors or warnings. | Status: not_done
- [ ] **Verify `npm run test` passes** — All vitest tests pass. | Status: not_done
- [ ] **Verify package exports** — `require('embed-cluster')` and `import { cluster } from 'embed-cluster'` both resolve correctly. Check `main` and `types` fields in package.json. | Status: not_done
- [ ] **Verify CLI binary** — `npx embed-cluster --help` prints usage. `npx embed-cluster --version` prints version. | Status: not_done

---

## Phase 23: Version Bump and Publishing Preparation

- [ ] **Bump version in package.json** — Follow semver based on the scope of changes. Phase 1 core = v0.1.0 (already set). Subsequent phases bump minor version as described in the roadmap (v0.2.0 through v1.0.0). | Status: not_done
- [ ] **Verify `files` field** — Ensure only `dist/` is published. No source files, test files, or fixtures in the published package. | Status: not_done
- [ ] **Verify `prepublishOnly` script** — Confirm `npm run build` runs automatically before `npm publish`. | Status: not_done
- [ ] **Add keywords to package.json** — Add relevant keywords: `embedding`, `cluster`, `kmeans`, `k-means`, `silhouette`, `tfidf`, `topic`, `nlp`, `ai`, `machine-learning`. | Status: not_done
- [ ] **Add description and author to package.json** — Fill in any missing metadata fields. | Status: not_done

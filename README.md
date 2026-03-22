# embed-cluster

Cluster embedding vectors into semantically coherent groups with automatic k selection, silhouette analysis, and custom labeling.

[![npm version](https://img.shields.io/npm/v/embed-cluster.svg)](https://www.npmjs.com/package/embed-cluster)
[![license](https://img.shields.io/npm/l/embed-cluster.svg)](https://github.com/SiluPanda/embed-cluster/blob/master/LICENSE)
[![node](https://img.shields.io/node/v/embed-cluster.svg)](https://nodejs.org)
[![types](https://img.shields.io/npm/types/embed-cluster.svg)](https://www.npmjs.com/package/embed-cluster)

---

## Description

`embed-cluster` groups high-dimensional embedding vectors into semantically coherent clusters using k-means++ initialization and configurable distance metrics. It is built for the characteristics of vectors produced by modern language model embedding APIs (768--3072 dimensions, cosine-similarity geometry) where generic clustering libraries require significant hand-tuning.

The package provides a complete clustering pipeline in a single function call: L2 normalization, k-means++ centroid initialization, iterative assignment and convergence, silhouette quality scoring, and optional automatic k selection. All algorithms are self-contained with zero mandatory runtime dependencies.

Key capabilities:

- **k-means++ clustering** with smart centroid initialization for faster convergence and better separation.
- **Automatic k selection** via silhouette analysis across a range of candidate k values.
- **Silhouette scoring** at the per-item, per-cluster, and aggregate level.
- **Custom labeling** through a caller-supplied sync or async labeling function.
- **Reproducible results** using a seeded pseudo-random number generator.
- **L2 normalization** built in, enabled by default, so cosine-distance clustering works out of the box.
- **Custom distance functions** for specialized similarity metrics beyond Euclidean and cosine.

---

## Installation

```bash
npm install embed-cluster
```

For optional dimensionality reduction, install the peer dependencies:

```bash
npm install ml-pca    # PCA-based dimensionality reduction
npm install umap-js   # UMAP-based dimensionality reduction
```

Both peer dependencies are optional and the core clustering API works without them.

---

## Quick Start

```ts
import { cluster } from "embed-cluster";

// Prepare items with id, text, and embedding vector
const items = [
  { id: "doc-1", text: "Introduction to machine learning", embedding: [0.12, 0.85, 0.33, ...] },
  { id: "doc-2", text: "Deep learning architectures",      embedding: [0.14, 0.82, 0.31, ...] },
  { id: "doc-3", text: "Cooking Italian pasta",             embedding: [0.91, 0.05, 0.72, ...] },
  // ...more items
];

// Cluster with a fixed k
const result = await cluster(items, { k: 3, seed: 42 });

console.log(result.k);             // 3
console.log(result.clusters);      // Array of 3 Cluster objects
console.log(result.quality);       // { silhouette, inertia }
console.log(result.converged);     // true
console.log(result.durationMs);    // elapsed time in milliseconds
```

### Automatic k selection

```ts
const result = await cluster(items, { autoK: true, maxK: 10 });
// k is chosen automatically to maximize silhouette score
console.log(result.k); // e.g. 4
```

### Pre-configured clusterer

```ts
import { createClusterer } from "embed-cluster";

const clusterer = createClusterer({
  autoK: true,
  maxK: 15,
  normalize: true,
  seed: 42,
});

const result = await clusterer.cluster(items);
const optimal = await clusterer.findOptimalK(items);
const quality = clusterer.silhouetteScore(result);
```

### Custom labeling

```ts
const result = await cluster(items, {
  k: 5,
  labeler: async (clusterItems, clusterId) => {
    // Call an LLM, run TF-IDF, or apply any labeling logic
    const texts = clusterItems.map((item) => item.text).join("\n");
    return `Topic ${clusterId}: ${texts.slice(0, 50)}`;
  },
});

for (const c of result.clusters) {
  console.log(c.label); // "Topic 0: Introduction to machine learning..."
}
```

---

## Features

- **k-means++ initialization** -- Selects initial centroids using D-squared weighted probabilistic sampling, producing better starting positions than random initialization and converging in fewer iterations.
- **Silhouette analysis** -- Computes per-item silhouette coefficients measuring how well each point fits its assigned cluster versus the nearest alternative cluster. Returns per-cluster and overall mean scores in the range [-1, 1].
- **Automatic k selection** -- Sweeps k from 2 to min(maxK, floor(sqrt(n))), scores each with silhouette analysis, and selects the k that maximizes the mean silhouette coefficient.
- **L2 normalization** -- Normalizes embedding vectors to unit length before clustering (enabled by default), which is the correct preprocessing for cosine-distance clustering of embedding vectors.
- **Custom distance functions** -- Supply any `(a: number[], b: number[]) => number` function to replace the default Euclidean distance. Built-in `cosineDistance` is exported for convenience.
- **Custom labeling** -- Provide a sync or async `LabelerFn` to generate human-readable labels for each cluster. The function receives the cluster's items and cluster ID.
- **Seeded PRNG** -- Deterministic pseudo-random number generation (mulberry32) ensures identical results across runs when a seed is provided.
- **Convergence control** -- Configurable maximum iterations and tolerance threshold for centroid movement. The result reports whether the algorithm converged and how many iterations were needed.
- **Quality metrics** -- Every result includes inertia (within-cluster sum of squared distances), silhouette scores, cohesion (average pairwise intra-cluster distance), and average distance to centroid per cluster.
- **Zero runtime dependencies** -- All algorithms are self-contained TypeScript. No mandatory third-party libraries.
- **TypeScript-first** -- Full type definitions with strict typing throughout. All interfaces and types are exported for consumer use.

---

## API Reference

### `cluster(items, options?)`

Cluster a set of `EmbedItem` objects using k-means++ and return a `ClusterResult` with silhouette scores.

```ts
function cluster(
  items: EmbedItem[],
  options?: ClusterOptions
): Promise<ClusterResult>;
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `items` | `EmbedItem[]` | Array of items to cluster. Must not be empty. |
| `options` | `ClusterOptions` | Clustering configuration. Provide `k` or set `autoK: true`. |

**Returns:** `Promise<ClusterResult>`

**Throws:** `ClusterError` with code `EMPTY_INPUT` if items is empty, `INVALID_OPTIONS` if neither `k` nor `autoK` is provided, `INVALID_K` if k is invalid, `INCONSISTENT_DIMENSIONS` if embedding dimensions differ.

---

### `createClusterer(config?)`

Create a pre-configured `Clusterer` instance. The returned object exposes `cluster()`, `findOptimalK()`, and `silhouetteScore()` methods that merge the bound config with per-call options.

```ts
function createClusterer(config?: ClusterOptions): Clusterer;
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ClusterOptions` | Default configuration applied to all method calls. Per-call options override these defaults. |

**Returns:** `Clusterer`

The `Clusterer` interface:

```ts
interface Clusterer {
  cluster(items: EmbedItem[], options?: ClusterOptions): Promise<ClusterResult>;
  findOptimalK(items: EmbedItem[], options?: Omit<ClusterOptions, "k">): Promise<OptimalKResult>;
  silhouetteScore(result: ClusterResult): SilhouetteResult;
}
```

---

### `findOptimalK(items, options?)`

Try k from 2 to min(maxK, floor(sqrt(n))), run k-means for each, compute the silhouette score, and return the k that maximizes it.

```ts
function findOptimalK(
  items: EmbedItem[],
  options?: Omit<ClusterOptions, "k">
): OptimalKResult;
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `items` | `EmbedItem[]` | Array of items to evaluate. |
| `options` | `Omit<ClusterOptions, "k">` | Configuration (k is excluded since it is being searched). |

**Returns:** `OptimalKResult`

```ts
interface OptimalKResult {
  k: number;                                                    // optimal k
  scores: Array<{ k: number; silhouette: number; inertia: number }>;  // score per candidate
  method: "silhouette" | "elbow" | "combined";                  // selection method used
}
```

---

### `silhouetteScore(result, distFn?)`

Compute the silhouette score for an existing `ClusterResult`.

```ts
function silhouetteScore(
  result: ClusterResult,
  distFn?: (a: number[], b: number[]) => number
): SilhouetteResult;
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `ClusterResult` | -- | A clustering result to evaluate. |
| `distFn` | `(a: number[], b: number[]) => number` | `euclideanDistance` | Distance function for silhouette computation. |

**Returns:** `SilhouetteResult`

```ts
interface SilhouetteResult {
  score: number;        // overall mean silhouette coefficient (-1 to 1)
  perCluster: number[]; // per-cluster mean silhouette
  perItem?: number[];   // per-item silhouette (optional, expensive)
}
```

Returns `{ score: 0, perCluster: [0, ...], perItem: [0, ...] }` when fewer than 2 clusters exist.

---

### `kMeans(items, k, options?)`

Low-level k-means implementation. Runs a single k-means pass with k-means++ initialization and returns a `ClusterResult` with placeholder silhouette scores (use `silhouetteScore()` separately to populate them).

```ts
function kMeans(
  items: EmbedItem[],
  k: number,
  options?: ClusterOptions
): ClusterResult;
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `items` | `EmbedItem[]` | Array of items to cluster. |
| `k` | `number` | Number of clusters. Must be a positive integer not exceeding `items.length`. |
| `options` | `ClusterOptions` | Clustering configuration. |

**Returns:** `ClusterResult`

---

### `kMeansPlusPlusInit(vectors, k, distFn, rand)`

k-means++ centroid initialization. Selects k initial centroids from the input vectors using D-squared weighted probabilistic selection.

```ts
function kMeansPlusPlusInit(
  vectors: number[][],
  k: number,
  distFn: (a: number[], b: number[]) => number,
  rand: () => number
): number[][];
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vectors` | `number[][]` | Input data points. |
| `k` | `number` | Number of centroids to select. |
| `distFn` | `(a: number[], b: number[]) => number` | Distance function. |
| `rand` | `() => number` | Random number generator returning values in [0, 1). |

**Returns:** `number[][]` -- Array of k centroid vectors.

---

### `euclideanDistance(a, b)`

Compute the Euclidean distance between two vectors.

```ts
function euclideanDistance(a: number[], b: number[]): number;
```

---

### `cosineDistance(a, b)`

Compute the cosine distance (1 - cosine similarity) between two vectors. Returns 1 for zero vectors.

```ts
function cosineDistance(a: number[], b: number[]): number;
```

---

### `normalizeVector(vec)`

L2-normalize a single vector to unit length. Returns a copy; does not mutate the input. Returns a zero vector unchanged.

```ts
function normalizeVector(vec: number[]): number[];
```

---

### `normalizeVectors(vecs)`

L2-normalize a batch of vectors independently.

```ts
function normalizeVectors(vecs: number[][]): number[][];
```

---

### `ClusterError`

Error class with a typed `code` field for programmatic error handling.

```ts
class ClusterError extends Error {
  readonly name: "ClusterError";
  readonly code: ClusterErrorCode;
  constructor(message: string, code: ClusterErrorCode);
}
```

---

## Types

All TypeScript interfaces are exported from the package entry point.

### `EmbedItem`

```ts
interface EmbedItem {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}
```

An input item pairing text content with its embedding vector. The optional `metadata` field carries arbitrary data through the clustering pipeline.

### `ClusterItem`

```ts
interface ClusterItem extends EmbedItem {
  clusterId: number;
  distanceToCentroid: number;
}
```

An `EmbedItem` after cluster assignment, annotated with the assigned cluster ID and its distance to the cluster centroid.

### `Cluster`

```ts
interface Cluster {
  id: number;
  centroid: number[];
  items: ClusterItem[];
  label?: string;
  size: number;
  avgDistanceToCentroid: number;
  cohesion: number; // average intra-cluster pairwise distance
}
```

A single cluster containing its centroid, assigned items, optional label, and quality metrics.

### `ClusterOptions`

```ts
interface ClusterOptions {
  k?: number;
  autoK?: boolean;
  maxK?: number;
  maxIterations?: number;
  tolerance?: number;
  seed?: number;
  normalize?: boolean;
  labeler?: LabelerFn;
  distanceFn?: (a: number[], b: number[]) => number;
}
```

### `ClusterResult`

```ts
interface ClusterResult {
  clusters: Cluster[];
  quality: ClusterQuality;
  k: number;
  iterations: number;
  converged: boolean;
  durationMs: number;
}
```

### `ClusterQuality`

```ts
interface ClusterQuality {
  silhouette: SilhouetteResult;
  inertia: number;
  daviesBouldin?: number;
  calinski?: number;
}
```

### `SilhouetteResult`

```ts
interface SilhouetteResult {
  score: number;        // overall mean (-1 to 1)
  perCluster: number[]; // per-cluster mean
  perItem?: number[];   // per-item scores
}
```

### `OptimalKResult`

```ts
interface OptimalKResult {
  k: number;
  scores: Array<{ k: number; silhouette: number; inertia: number }>;
  method: "silhouette" | "elbow" | "combined";
}
```

### `VisualizationData`

```ts
interface VisualizationData {
  points: Array<{ id: string; x: number; y: number; clusterId: number }>;
  method: "pca" | "umap" | "tsne";
}
```

### `LabelerFn`

```ts
type LabelerFn = (
  items: EmbedItem[],
  clusterId: number
) => Promise<string> | string;
```

### `ClusterErrorCode`

```ts
type ClusterErrorCode =
  | "EMPTY_INPUT"
  | "INCONSISTENT_DIMENSIONS"
  | "DEGENERATE_INPUT"
  | "INVALID_K"
  | "INVALID_OPTIONS";
```

---

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `k` | `number` | -- | Number of clusters. Required when `autoK` is `false`. |
| `autoK` | `boolean` | `false` | Automatically select the optimal k using silhouette analysis. |
| `maxK` | `number` | `min(10, floor(sqrt(n)))` | Maximum k to evaluate when `autoK` is `true`. |
| `maxIterations` | `number` | `100` | Maximum number of k-means iterations before stopping. |
| `tolerance` | `number` | `1e-4` | Convergence tolerance. The algorithm stops when the maximum centroid shift falls below this value. |
| `seed` | `number` | `42` | Seed for the pseudo-random number generator. Set to any integer for reproducible results. |
| `normalize` | `boolean` | `true` | L2-normalize all embedding vectors before clustering. Recommended for cosine-distance semantics. |
| `labeler` | `LabelerFn` | -- | Custom function to generate a label for each cluster. Called once per cluster after assignment. |
| `distanceFn` | `(a: number[], b: number[]) => number` | `euclideanDistance` | Custom distance function. Use `cosineDistance` for angular separation or provide your own metric. |

---

## Error Handling

All errors thrown by the library are instances of `ClusterError`, which extends `Error` and carries a typed `code` field for programmatic handling.

```ts
import { cluster, ClusterError } from "embed-cluster";

try {
  await cluster([], { k: 3 });
} catch (err) {
  if (err instanceof ClusterError) {
    switch (err.code) {
      case "EMPTY_INPUT":
        console.error("No items provided");
        break;
      case "INVALID_K":
        console.error("k is out of range");
        break;
      case "INVALID_OPTIONS":
        console.error("Provide k or set autoK: true");
        break;
      case "INCONSISTENT_DIMENSIONS":
        console.error("All embeddings must have the same dimension");
        break;
      case "DEGENERATE_INPUT":
        console.error("Input data is degenerate");
        break;
    }
  }
}
```

| Error Code | Condition |
|------------|-----------|
| `EMPTY_INPUT` | The items array is empty. |
| `INCONSISTENT_DIMENSIONS` | Embedding vectors have different lengths. |
| `DEGENERATE_INPUT` | Input data is degenerate (e.g., all identical vectors). |
| `INVALID_K` | k is not a positive integer, or k exceeds the number of items. |
| `INVALID_OPTIONS` | Neither `k` nor `autoK: true` was provided. |

---

## Advanced Usage

### Using cosine distance

```ts
import { cluster, cosineDistance } from "embed-cluster";

const result = await cluster(items, {
  k: 5,
  normalize: true,
  distanceFn: cosineDistance,
});
```

When `normalize` is `true` (the default), all vectors are L2-normalized before clustering. Combined with `cosineDistance`, this performs angular clustering -- the standard approach for embedding vectors from language models.

### Evaluating an existing clustering

```ts
import { kMeans, silhouetteScore, cosineDistance } from "embed-cluster";

const result = kMeans(items, 4, { seed: 123, normalize: true });
const quality = silhouetteScore(result, cosineDistance);

console.log(quality.score);      // overall mean silhouette
console.log(quality.perCluster); // [0.72, 0.85, 0.61, 0.78]
console.log(quality.perItem);    // per-item scores (same length as items)
```

### Comparing different k values

```ts
import { findOptimalK } from "embed-cluster";

const optimal = findOptimalK(items, {
  maxK: 15,
  normalize: true,
  seed: 42,
});

console.log(`Best k: ${optimal.k}`);
for (const entry of optimal.scores) {
  console.log(`  k=${entry.k}  silhouette=${entry.silhouette.toFixed(3)}  inertia=${entry.inertia.toFixed(1)}`);
}
```

### Identifying outliers

Points with a negative per-item silhouette coefficient are poorly assigned and may be outliers:

```ts
const result = await cluster(items, { k: 5, seed: 42 });
const sil = silhouetteScore(result);

const outlierIndices: number[] = [];
if (sil.perItem) {
  sil.perItem.forEach((score, index) => {
    if (score < 0) {
      outlierIndices.push(index);
    }
  });
}
console.log(`Found ${outlierIndices.length} outlier(s)`);
```

### Extracting cluster membership

```ts
const result = await cluster(items, { k: 4, seed: 42 });

for (const c of result.clusters) {
  console.log(`Cluster ${c.id} (${c.size} items, cohesion=${c.cohesion.toFixed(4)}):`);
  for (const item of c.items) {
    console.log(`  ${item.id}: ${item.text} (dist=${item.distanceToCentroid.toFixed(4)})`);
  }
}
```

### Reusing configuration across calls

```ts
import { createClusterer } from "embed-cluster";

const clusterer = createClusterer({
  normalize: true,
  seed: 42,
  maxIterations: 200,
  tolerance: 1e-6,
});

// All calls inherit the bound configuration
const r1 = await clusterer.cluster(datasetA, { k: 3 });
const r2 = await clusterer.cluster(datasetB, { k: 5 });

// Per-call options override bound config
const r3 = await clusterer.cluster(datasetC, { autoK: true, maxK: 20 });
```

---

## TypeScript

The package is written in TypeScript with strict mode enabled and ships type declarations alongside the compiled JavaScript. All interfaces, types, and the `ClusterError` class are exported from the package entry point.

```ts
import type {
  EmbedItem,
  ClusterItem,
  Cluster,
  ClusterOptions,
  ClusterResult,
  ClusterQuality,
  SilhouetteResult,
  OptimalKResult,
  VisualizationData,
  LabelerFn,
  Clusterer,
  ClusterErrorCode,
} from "embed-cluster";
```

The package targets ES2022 and compiles to CommonJS. It requires Node.js 18 or later.

---

## License

MIT

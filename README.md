# embed-cluster

Cluster embeddings into topics with automatic labeling.

## Installation

```bash
npm install embed-cluster
```

For dimensionality reduction (PCA or UMAP), install the optional peer dependencies:

```bash
npm install ml-pca    # PCA-based dimensionality reduction
npm install umap-js   # UMAP-based dimensionality reduction
```

## Quick Start

```ts
import { cluster, createClusterer } from 'embed-cluster';

// Cluster raw embedding vectors
const result = await cluster(embeddings, { k: 5 });

console.log(result.clusters); // 5 clusters with labels, centroids, items
console.log(result.quality);  // silhouette score, inertia, interpretation

// Or use a pre-configured clusterer
const clusterer = createClusterer({
  autoK: true,
  maxK: 15,
  normalize: true,
});

const result = await clusterer.cluster(items);
const optimal = await clusterer.findOptimalK(items);
```

## Available Exports

### Functions

- **`normalizeVector(vec: number[]): number[]`** -- L2-normalize a single vector to unit length. Returns a zero vector unchanged.
- **`normalizeVectors(vecs: number[][]): number[][]`** -- L2-normalize a batch of vectors independently.

### Types

All TypeScript interfaces are exported for consumer use:

- `EmbedItem` -- Input item with id, text, embedding, and optional metadata
- `ClusterItem` -- An `EmbedItem` assigned to a cluster with distance-to-centroid
- `Cluster` -- A cluster with centroid, items, label, size, and cohesion metrics
- `ClusterOptions` -- Configuration for clustering (k, autoK, maxK, tolerance, seed, etc.)
- `ClusterResult` -- Full result including clusters, quality metrics, iteration count, and timing
- `Clusterer` -- Interface for a pre-configured clusterer instance
- `SilhouetteResult` -- Silhouette coefficient scores (overall, per-cluster, per-item)
- `OptimalKResult` -- Result of automatic k selection with scores per candidate k
- `ClusterQuality` -- Quality metrics including silhouette, inertia, and optional indices
- `VisualizationData` -- 2D projected points for visualization (PCA, UMAP, or t-SNE)
- `LabelerFn` -- Custom labeling function signature

### Error Handling

- **`ClusterError`** -- Error class with typed `code` field for programmatic error handling
- **`ClusterErrorCode`** -- Union type of error codes: `EMPTY_INPUT`, `INCONSISTENT_DIMENSIONS`, `DEGENERATE_INPUT`, `INVALID_K`, `INVALID_OPTIONS`

```ts
import { ClusterError } from 'embed-cluster';

try {
  const result = await cluster([], { k: 3 });
} catch (err) {
  if (err instanceof ClusterError) {
    console.error(err.code); // 'EMPTY_INPUT'
  }
}
```

## Features

- **k-means++ clustering** -- Smart centroid initialization for faster convergence and better results
- **Silhouette analysis** -- Evaluate cluster quality with per-point and per-cluster silhouette coefficients
- **Automatic k selection** -- Find the optimal number of clusters using silhouette and elbow methods
- **PCA/UMAP visualization** -- Reduce high-dimensional embeddings to 2D for visualization (requires optional peer deps)
- **L2 normalization** -- Built-in vector normalization for cosine-distance clustering
- **Custom labeling** -- Provide your own labeling function or use built-in TF-IDF topic extraction
- **Reproducible results** -- Seed-based random number generation for deterministic clustering
- **TypeScript-first** -- Full type definitions with strict typing throughout

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `k` | `number` | -- | Number of clusters (required if `autoK` is false) |
| `autoK` | `boolean` | `false` | Automatically select optimal k |
| `maxK` | `number` | `10` | Maximum k to try when `autoK` is true |
| `maxIterations` | `number` | `100` | Maximum k-means iterations |
| `tolerance` | `number` | `1e-4` | Convergence tolerance |
| `seed` | `number` | -- | Random seed for reproducibility |
| `normalize` | `boolean` | `true` | L2-normalize embeddings before clustering |
| `labeler` | `LabelerFn` | -- | Custom labeling function |
| `distanceFn` | `Function` | -- | Custom distance function |

## CLI

```bash
npx embed-cluster --input embeddings.json --k 5 --format summary
```

## License

MIT

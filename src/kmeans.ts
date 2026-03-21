import { ClusterError } from './errors';
import type { EmbedItem, ClusterItem, Cluster, ClusterResult, ClusterOptions, ClusterQuality } from './types';
import { normalizeVectors } from './normalize';

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

export function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export function cosineDistance(a: number[], b: number[]): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  if (denom === 0) return 1;
  return 1 - dot / denom;
}

// ---------------------------------------------------------------------------
// Seeded pseudo-random number generator (mulberry32)
// ---------------------------------------------------------------------------

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// k-means++ initialization
// ---------------------------------------------------------------------------

export function kMeansPlusPlusInit(
  vectors: number[][],
  k: number,
  distFn: (a: number[], b: number[]) => number,
  rand: () => number,
): number[][] {
  const n = vectors.length;
  const centroids: number[][] = [];

  // Pick first centroid uniformly at random
  const firstIdx = Math.floor(rand() * n);
  centroids.push([...vectors[firstIdx]]);

  for (let c = 1; c < k; c++) {
    // Compute distance squared from each point to its nearest centroid
    const distances = vectors.map(v => {
      let minDist = Infinity;
      for (const centroid of centroids) {
        const d = distFn(v, centroid);
        if (d < minDist) minDist = d;
      }
      return minDist * minDist;
    });

    // Weighted random selection proportional to distance squared
    const totalWeight = distances.reduce((s, d) => s + d, 0);
    let threshold = rand() * totalWeight;
    let chosen = n - 1;
    for (let i = 0; i < n; i++) {
      threshold -= distances[i];
      if (threshold <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push([...vectors[chosen]]);
  }

  return centroids;
}

// ---------------------------------------------------------------------------
// Centroid computation
// ---------------------------------------------------------------------------

function computeCentroid(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0].length;
  const centroid = new Array<number>(dim).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) {
      centroid[i] += v[i];
    }
  }
  for (let i = 0; i < dim; i++) {
    centroid[i] /= vectors.length;
  }
  return centroid;
}

function centroidShift(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// ---------------------------------------------------------------------------
// Inertia (sum of squared distances to assigned centroid)
// ---------------------------------------------------------------------------

function computeInertia(
  assignments: number[],
  vectors: number[][],
  centroids: number[][],
  distFn: (a: number[], b: number[]) => number,
): number {
  let inertia = 0;
  for (let i = 0; i < vectors.length; i++) {
    const d = distFn(vectors[i], centroids[assignments[i]]);
    inertia += d * d;
  }
  return inertia;
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function validateInput(items: EmbedItem[], k: number): void {
  if (items.length === 0) {
    throw new ClusterError('Input must not be empty', 'EMPTY_INPUT');
  }
  const dim = items[0].embedding.length;
  for (const item of items) {
    if (item.embedding.length !== dim) {
      throw new ClusterError('All embeddings must have the same dimension', 'INCONSISTENT_DIMENSIONS');
    }
  }
  if (k < 1 || !Number.isInteger(k)) {
    throw new ClusterError('k must be a positive integer', 'INVALID_K');
  }
  if (k > items.length) {
    throw new ClusterError(`k (${k}) cannot exceed number of items (${items.length})`, 'INVALID_K');
  }
}

// ---------------------------------------------------------------------------
// Main k-means function
// ---------------------------------------------------------------------------

export function kMeans(items: EmbedItem[], k: number, options: ClusterOptions = {}): ClusterResult {
  const startMs = Date.now();

  validateInput(items, k);

  const {
    maxIterations = 100,
    tolerance = 1e-4,
    seed = 42,
    normalize = true,
    distanceFn,
  } = options;

  const distFn = distanceFn ?? euclideanDistance;
  const rand = makePrng(seed);

  // Prepare vectors (optionally normalize)
  let vectors = items.map(it => it.embedding);
  if (normalize) {
    vectors = normalizeVectors(vectors);
  }

  // Initialize centroids with k-means++
  let centroids = kMeansPlusPlusInit(vectors, k, distFn, rand);

  let assignments = new Array<number>(vectors.length).fill(0);
  let iterations = 0;
  let converged = false;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations++;

    // Assignment step
    const newAssignments = vectors.map(v => {
      let minDist = Infinity;
      let best = 0;
      for (let c = 0; c < k; c++) {
        const d = distFn(v, centroids[c]);
        if (d < minDist) {
          minDist = d;
          best = c;
        }
      }
      return best;
    });

    // Update step — recompute centroids
    const clusterVectors: number[][][] = Array.from({ length: k }, () => []);
    for (let i = 0; i < vectors.length; i++) {
      clusterVectors[newAssignments[i]].push(vectors[i]);
    }

    const newCentroids = centroids.map((old, c) => {
      if (clusterVectors[c].length === 0) {
        // Empty cluster: keep old centroid
        return old;
      }
      return computeCentroid(clusterVectors[c]);
    });

    // Check convergence
    const maxShift = newCentroids.reduce((max, nc, c) => {
      return Math.max(max, centroidShift(nc, centroids[c]));
    }, 0);

    assignments = newAssignments;
    centroids = newCentroids;

    if (maxShift < tolerance) {
      converged = true;
      break;
    }
  }

  // Build Cluster objects
  const clusterItems: ClusterItem[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < vectors.length; i++) {
    const cid = assignments[i];
    const dist = distFn(vectors[i], centroids[cid]);
    clusterItems[cid].push({
      ...items[i],
      embedding: vectors[i],   // use the (possibly normalized) vector
      clusterId: cid,
      distanceToCentroid: dist,
    });
  }

  const clusters: Cluster[] = centroids.map((centroid, c) => {
    const members = clusterItems[c];
    const avgDist = members.length > 0
      ? members.reduce((s, m) => s + m.distanceToCentroid, 0) / members.length
      : 0;

    // Cohesion: average pairwise distance within cluster
    let cohesion = 0;
    if (members.length > 1) {
      let pairSum = 0;
      let pairCount = 0;
      for (let a = 0; a < members.length; a++) {
        for (let b = a + 1; b < members.length; b++) {
          pairSum += distFn(members[a].embedding, members[b].embedding);
          pairCount++;
        }
      }
      cohesion = pairCount > 0 ? pairSum / pairCount : 0;
    }

    return {
      id: c,
      centroid,
      items: members,
      size: members.length,
      avgDistanceToCentroid: avgDist,
      cohesion,
    };
  });

  const inertia = computeInertia(assignments, vectors, centroids, distFn);

  // Silhouette scores are computed separately in silhouette.ts
  // Provide a placeholder here; callers can compute via silhouetteScore()
  const quality: ClusterQuality = {
    silhouette: { score: 0, perCluster: new Array(k).fill(0) },
    inertia,
  };

  return {
    clusters,
    quality,
    k,
    iterations,
    converged,
    durationMs: Date.now() - startMs,
  };
}

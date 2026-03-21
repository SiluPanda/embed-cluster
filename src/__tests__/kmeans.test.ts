import { describe, it, expect } from 'vitest';
import { kMeans, euclideanDistance, cosineDistance, kMeansPlusPlusInit } from '../kmeans';
import type { EmbedItem } from '../types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeItems(coords: number[][]): EmbedItem[] {
  return coords.map((embedding, i) => ({ id: `item-${i}`, text: `doc ${i}`, embedding }));
}

// Two clearly separated 2D clusters
const GROUP_A = [[1, 0], [1.1, 0.1], [0.9, -0.1], [1.05, 0.05]];
const GROUP_B = [[-1, 0], [-1.1, 0.1], [-0.9, -0.1], [-1.05, 0.05]];

const TWO_CLUSTER_ITEMS = makeItems([...GROUP_A, ...GROUP_B]);

// Three clearly separated 2D clusters
const GROUP_C = [[0, 1], [0.1, 1.1], [-0.1, 0.9]];
const THREE_CLUSTER_ITEMS = makeItems([...GROUP_A, ...GROUP_B, ...GROUP_C]);

// ---------------------------------------------------------------------------
// euclideanDistance
// ---------------------------------------------------------------------------

describe('euclideanDistance', () => {
  it('returns 0 for identical vectors', () => {
    expect(euclideanDistance([1, 2, 3], [1, 2, 3])).toBeCloseTo(0, 10);
  });

  it('returns 5 for [0,0] vs [3,4]', () => {
    expect(euclideanDistance([0, 0], [3, 4])).toBeCloseTo(5, 10);
  });

  it('is symmetric', () => {
    const a = [1, 2, 3];
    const b = [4, 5, 6];
    expect(euclideanDistance(a, b)).toBeCloseTo(euclideanDistance(b, a), 10);
  });
});

// ---------------------------------------------------------------------------
// cosineDistance
// ---------------------------------------------------------------------------

describe('cosineDistance', () => {
  it('returns 0 for identical unit vectors', () => {
    expect(cosineDistance([1, 0], [1, 0])).toBeCloseTo(0, 10);
  });

  it('returns 1 for orthogonal vectors', () => {
    expect(cosineDistance([1, 0], [0, 1])).toBeCloseTo(1, 10);
  });

  it('returns 2 for opposite vectors', () => {
    expect(cosineDistance([1, 0], [-1, 0])).toBeCloseTo(2, 10);
  });

  it('returns 1 for zero vector (degenerate case)', () => {
    expect(cosineDistance([0, 0], [1, 0])).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// kMeansPlusPlusInit
// ---------------------------------------------------------------------------

describe('kMeansPlusPlusInit', () => {
  it('returns k centroids', () => {
    const vectors = makeItems([[1, 0], [0, 1], [-1, 0], [0, -1]]).map(i => i.embedding);
    const rand = () => 0.5; // deterministic-ish
    const centroids = kMeansPlusPlusInit(vectors, 2, euclideanDistance, rand);
    expect(centroids).toHaveLength(2);
  });

  it('each centroid is a copy of an input vector', () => {
    const vectors = makeItems([[1, 0], [0, 1], [-1, 0]]).map(i => i.embedding);
    const rand = () => 0.3;
    const centroids = kMeansPlusPlusInit(vectors, 2, euclideanDistance, rand);
    for (const c of centroids) {
      const match = vectors.some(v => v.every((val, i) => val === c[i]));
      expect(match).toBe(true);
    }
  });

  it('centroids are distinct for well-separated data', () => {
    const vectors = [...GROUP_A, ...GROUP_B];
    const rand = (() => { let i = 0; const seq = [0.1, 0.9]; return () => seq[i++ % seq.length]; })();
    const centroids = kMeansPlusPlusInit(vectors, 2, euclideanDistance, rand);
    expect(centroids).toHaveLength(2);
    // They should not be the same point
    const [c0, c1] = centroids;
    const same = c0.every((v, i) => v === c1[i]);
    expect(same).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// kMeans
// ---------------------------------------------------------------------------

describe('kMeans', () => {
  it('returns k clusters for k=2', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    expect(result.clusters).toHaveLength(2);
    expect(result.k).toBe(2);
  });

  it('all items are assigned to some cluster', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    const totalItems = result.clusters.reduce((s, c) => s + c.items.length, 0);
    expect(totalItems).toBe(TWO_CLUSTER_ITEMS.length);
  });

  it('converges for well-separated clusters', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    expect(result.converged).toBe(true);
  });

  it('centroids are near the true group means for k=2', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    // Sort clusters by centroid x coordinate
    const sorted = [...result.clusters].sort((a, b) => a.centroid[0] - b.centroid[0]);
    // Negative-x cluster centroid should be near -1
    expect(sorted[0].centroid[0]).toBeCloseTo(-1.0125, 1);
    // Positive-x cluster centroid should be near +1
    expect(sorted[1].centroid[0]).toBeCloseTo(1.0375, 1);
  });

  it('assigns GROUP_A items to one cluster and GROUP_B to another (k=2)', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    // All items in GROUP_A should share a cluster, all in GROUP_B another
    const sorted = [...result.clusters].sort((a, b) => a.centroid[0] - b.centroid[0]);
    const negCluster = sorted[0];
    const posCluster = sorted[1];
    for (const item of negCluster.items) {
      expect(item.id.startsWith('item-4') || parseInt(item.id.split('-')[1]) >= 4).toBe(true);
    }
    for (const item of posCluster.items) {
      expect(parseInt(item.id.split('-')[1])).toBeLessThan(4);
    }
  });

  it('handles k=1 (single cluster contains all items)', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 1, { normalize: false, seed: 1 });
    expect(result.clusters).toHaveLength(1);
    expect(result.clusters[0].items).toHaveLength(TWO_CLUSTER_ITEMS.length);
  });

  it('handles k=3 on 3-cluster data', () => {
    const result = kMeans(THREE_CLUSTER_ITEMS, 3, { normalize: false, seed: 1 });
    expect(result.clusters).toHaveLength(3);
    const totalItems = result.clusters.reduce((s, c) => s + c.items.length, 0);
    expect(totalItems).toBe(THREE_CLUSTER_ITEMS.length);
  });

  it('inertia is finite and non-negative', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    expect(result.quality.inertia).toBeGreaterThanOrEqual(0);
    expect(isFinite(result.quality.inertia)).toBe(true);
  });

  it('inertia decreases as k increases', () => {
    const r1 = kMeans(THREE_CLUSTER_ITEMS, 1, { normalize: false, seed: 1 });
    const r3 = kMeans(THREE_CLUSTER_ITEMS, 3, { normalize: false, seed: 1 });
    expect(r3.quality.inertia).toBeLessThan(r1.quality.inertia);
  });

  it('cluster.size equals cluster.items.length', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    for (const c of result.clusters) {
      expect(c.size).toBe(c.items.length);
    }
  });

  it('durationMs is non-negative', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('is deterministic with same seed', () => {
    const r1 = kMeans(THREE_CLUSTER_ITEMS, 3, { normalize: false, seed: 42 });
    const r2 = kMeans(THREE_CLUSTER_ITEMS, 3, { normalize: false, seed: 42 });
    for (let c = 0; c < 3; c++) {
      expect(r1.clusters[c].centroid).toEqual(r2.clusters[c].centroid);
    }
  });

  it('supports custom distanceFn (cosine)', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, {
      normalize: true,
      seed: 1,
      distanceFn: cosineDistance,
    });
    expect(result.clusters).toHaveLength(2);
    expect(result.converged).toBe(true);
  });

  it('throws EMPTY_INPUT for empty array', () => {
    expect(() => kMeans([], 2)).toThrowError();
    try { kMeans([], 2); } catch (e: unknown) { expect((e as { code?: string }).code).toBe('EMPTY_INPUT'); }
  });

  it('throws INVALID_K when k > n', () => {
    expect(() => kMeans(makeItems([[1, 0]]), 2)).toThrowError();
    try { kMeans(makeItems([[1, 0]]), 2); } catch (e: unknown) { expect((e as { code?: string }).code).toBe('INVALID_K'); }
  });

  it('throws INCONSISTENT_DIMENSIONS for mismatched embeddings', () => {
    const items: EmbedItem[] = [
      { id: 'a', text: 'a', embedding: [1, 0] },
      { id: 'b', text: 'b', embedding: [1, 0, 0] },
    ];
    expect(() => kMeans(items, 2)).toThrowError();
    try { kMeans(items, 2); } catch (e: unknown) { expect((e as { code?: string }).code).toBe('INCONSISTENT_DIMENSIONS'); }
  });

  it('normalize=true does not change number of clusters', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: true, seed: 1 });
    expect(result.clusters).toHaveLength(2);
  });

  it('each ClusterItem has distanceToCentroid >= 0', () => {
    const result = kMeans(TWO_CLUSTER_ITEMS, 2, { normalize: false, seed: 1 });
    for (const c of result.clusters) {
      for (const item of c.items) {
        expect(item.distanceToCentroid).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('each ClusterItem.clusterId matches its parent cluster.id', () => {
    const result = kMeans(THREE_CLUSTER_ITEMS, 3, { normalize: false, seed: 1 });
    for (const c of result.clusters) {
      for (const item of c.items) {
        expect(item.clusterId).toBe(c.id);
      }
    }
  });
});

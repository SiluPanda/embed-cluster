import { describe, it, expect } from 'vitest';
import { silhouetteScore } from '../silhouette';
import { kMeans } from '../kmeans';
import type { EmbedItem, ClusterResult, Cluster, ClusterItem, ClusterQuality } from '../types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeItems(coords: number[][]): EmbedItem[] {
  return coords.map((embedding, i) => ({ id: `item-${i}`, text: `doc ${i}`, embedding }));
}

// Well-separated 2D clusters
const GROUP_A = [[1, 0], [1.1, 0.05], [0.9, -0.05], [1.05, 0.02]];
const GROUP_B = [[-1, 0], [-1.1, 0.05], [-0.9, -0.05], [-1.05, 0.02]];
const WELL_SEPARATED = makeItems([...GROUP_A, ...GROUP_B]);

// Mixed/overlapping clusters
const GROUP_MIXED = [[0, 0], [0.1, 0.1], [-0.1, -0.1], [0.05, -0.05]];
const GROUP_MIXED2 = [[0.2, 0.2], [0.3, 0.1], [0.1, 0.3], [0.25, 0.25]];
const MIXED = makeItems([...GROUP_MIXED, ...GROUP_MIXED2]);

// ---------------------------------------------------------------------------
// Helpers to build a synthetic ClusterResult for unit-testing silhouette
// ---------------------------------------------------------------------------

function buildClusterResult(
  groups: number[][][],
  baseItems: EmbedItem[],
): ClusterResult {
  let itemIndex = 0;
  const clusters: Cluster[] = groups.map((group, cid) => {
    const items: ClusterItem[] = group.map((embedding) => {
      const base = baseItems[itemIndex++];
      return { ...base, embedding, clusterId: cid, distanceToCentroid: 0 };
    });
    const centroid =
      group.length > 0
        ? group[0].map((_, i) => group.reduce((s, v) => s + v[i], 0) / group.length)
        : [];
    return { id: cid, centroid, items, size: items.length, avgDistanceToCentroid: 0, cohesion: 0 };
  });

  const quality: ClusterQuality = {
    silhouette: { score: 0, perCluster: [] },
    inertia: 0,
  };

  return { clusters, quality, k: groups.length, iterations: 1, converged: true, durationMs: 0 };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('silhouetteScore', () => {
  it('returns score in [-1, 1] for well-separated clusters', () => {
    const result = kMeans(WELL_SEPARATED, 2, { normalize: false, seed: 1 });
    const sil = silhouetteScore(result);
    expect(sil.score).toBeGreaterThanOrEqual(-1);
    expect(sil.score).toBeLessThanOrEqual(1);
  });

  it('well-separated clusters produce score > 0.5', () => {
    const result = kMeans(WELL_SEPARATED, 2, { normalize: false, seed: 1 });
    const sil = silhouetteScore(result);
    expect(sil.score).toBeGreaterThan(0.5);
  });

  it('perCluster array has length equal to k', () => {
    const result = kMeans(WELL_SEPARATED, 2, { normalize: false, seed: 1 });
    const sil = silhouetteScore(result);
    expect(sil.perCluster).toHaveLength(2);
  });

  it('perItem array has length equal to total items', () => {
    const result = kMeans(WELL_SEPARATED, 2, { normalize: false, seed: 1 });
    const sil = silhouetteScore(result);
    expect(sil.perItem).toHaveLength(WELL_SEPARATED.length);
  });

  it('returns score=0 for k=1 (single cluster)', () => {
    const result = kMeans(WELL_SEPARATED, 1, { normalize: false, seed: 1 });
    const sil = silhouetteScore(result);
    expect(sil.score).toBe(0);
  });

  it('each perItem value is in [-1, 1]', () => {
    const result = kMeans(WELL_SEPARATED, 2, { normalize: false, seed: 1 });
    const sil = silhouetteScore(result);
    for (const s of sil.perItem ?? []) {
      expect(s).toBeGreaterThanOrEqual(-1);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it('mixed clusters produce lower score than well-separated clusters', () => {
    const wellResult = kMeans(WELL_SEPARATED, 2, { normalize: false, seed: 1 });
    const mixedResult = kMeans(MIXED, 2, { normalize: false, seed: 1 });
    const wellSil = silhouetteScore(wellResult);
    const mixedSil = silhouetteScore(mixedResult);
    expect(wellSil.score).toBeGreaterThan(mixedSil.score);
  });

  it('score is the mean of perItem values', () => {
    const result = kMeans(WELL_SEPARATED, 2, { normalize: false, seed: 1 });
    const sil = silhouetteScore(result);
    const perItem = sil.perItem ?? [];
    const mean = perItem.reduce((s, v) => s + v, 0) / perItem.length;
    expect(sil.score).toBeCloseTo(mean, 10);
  });

  it('returns score=0 for result with zero clusters', () => {
    const emptyResult: ClusterResult = {
      clusters: [],
      quality: { silhouette: { score: 0, perCluster: [] }, inertia: 0 },
      k: 0,
      iterations: 0,
      converged: true,
      durationMs: 0,
    };
    const sil = silhouetteScore(emptyResult);
    expect(sil.score).toBe(0);
  });

  it('singleton clusters: items with no same-cluster neighbors have a=0', () => {
    // Build result with two singleton clusters
    const dummyItems = makeItems([[1, 0], [-1, 0]]);
    const result = buildClusterResult([[[1, 0]], [[-1, 0]]], dummyItems);
    const sil = silhouetteScore(result);
    // a = 0 for singletons, b = distance to the other cluster
    // s = (b - 0) / max(0, b) = 1
    expect(sil.perItem?.[0]).toBeCloseTo(1, 5);
    expect(sil.perItem?.[1]).toBeCloseTo(1, 5);
  });

  it('score improves with better clustering on 3-cluster data', () => {
    const GROUP_C = [[0, 1], [0.05, 1.1], [-0.05, 0.9]];
    const items3 = makeItems([...GROUP_A, ...GROUP_B, ...GROUP_C]);
    const r3 = kMeans(items3, 3, { normalize: false, seed: 1 });
    const r2 = kMeans(items3, 2, { normalize: false, seed: 1 });
    const sil3 = silhouetteScore(r3);
    const sil2 = silhouetteScore(r2);
    // With 3 actual clusters, k=3 should score better than k=2
    expect(sil3.score).toBeGreaterThan(sil2.score);
  });
});

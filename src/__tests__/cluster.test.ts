import { describe, it, expect } from 'vitest';
import { cluster, createClusterer, findOptimalK, silhouetteScore } from '../clusterer';
import { silhouetteScore as silhouetteScoreFromModule } from '../silhouette';
import { ClusterError } from '../errors';
import fixture from './fixtures/embeddings-small.json';
import type { EmbedItem } from '../types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeItems(coords: number[][]): EmbedItem[] {
  return coords.map((embedding, i) => ({ id: `item-${i}`, text: `doc ${i}`, embedding }));
}

// Two clear clusters: group near [1,0] and group near [-1,0]
const GROUP_A = [[1, 0.05], [1.1, 0], [0.9, -0.05], [1.0, 0.02]];
const GROUP_B = [[-1, 0.05], [-1.1, 0], [-0.9, -0.05], [-1.0, 0.02]];
const ITEMS_2 = makeItems([...GROUP_A, ...GROUP_B]);

// Three clear clusters
const GROUP_C = [[0, 1.0], [0.05, 1.1], [-0.05, 0.95]];
const ITEMS_3 = makeItems([...GROUP_A, ...GROUP_B, ...GROUP_C]);

// ---------------------------------------------------------------------------
// cluster()
// ---------------------------------------------------------------------------

describe('cluster()', () => {
  it('returns a ClusterResult with k=2 clusters', async () => {
    const result = await cluster(ITEMS_2, { k: 2, normalize: false, seed: 1 });
    expect(result.k).toBe(2);
    expect(result.clusters).toHaveLength(2);
  });

  it('all items appear in result', async () => {
    const result = await cluster(ITEMS_2, { k: 2, normalize: false, seed: 1 });
    const total = result.clusters.reduce((s, c) => s + c.items.length, 0);
    expect(total).toBe(ITEMS_2.length);
  });

  it('quality.silhouette.score is populated (non-zero for separated data)', async () => {
    const result = await cluster(ITEMS_2, { k: 2, normalize: false, seed: 1 });
    expect(result.quality.silhouette.score).toBeGreaterThan(0.5);
  });

  it('quality.silhouette.perCluster has length k', async () => {
    const result = await cluster(ITEMS_2, { k: 2, normalize: false, seed: 1 });
    expect(result.quality.silhouette.perCluster).toHaveLength(2);
  });

  it('throws ClusterError EMPTY_INPUT for empty array', async () => {
    await expect(cluster([], { k: 2 })).rejects.toThrow(ClusterError);
    await expect(cluster([], { k: 2 })).rejects.toMatchObject({ code: 'EMPTY_INPUT' });
  });

  it('throws ClusterError INVALID_OPTIONS when neither k nor autoK provided', async () => {
    await expect(cluster(ITEMS_2, {})).rejects.toThrow(ClusterError);
    await expect(cluster(ITEMS_2, {})).rejects.toMatchObject({ code: 'INVALID_OPTIONS' });
  });

  it('autoK=true selects k and returns result', async () => {
    const result = await cluster(ITEMS_3, { autoK: true, normalize: false, seed: 1 });
    expect(result.k).toBeGreaterThanOrEqual(2);
    expect(result.clusters).toHaveLength(result.k);
  });

  it('durationMs is populated', async () => {
    const result = await cluster(ITEMS_2, { k: 2, normalize: false, seed: 1 });
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('labeler is called for each cluster', async () => {
    const labels: string[] = [];
    const result = await cluster(ITEMS_2, {
      k: 2,
      normalize: false,
      seed: 1,
      labeler: (_items, id) => {
        const label = `cluster-${id}`;
        labels.push(label);
        return label;
      },
    });
    expect(labels).toHaveLength(2);
    for (const c of result.clusters) {
      expect(c.label).toBeDefined();
      expect(c.label).toMatch(/^cluster-\d$/);
    }
  });

  it('async labeler is supported', async () => {
    const result = await cluster(ITEMS_2, {
      k: 2,
      normalize: false,
      seed: 1,
      labeler: async (_items, id) => {
        return `async-label-${id}`;
      },
    });
    for (const c of result.clusters) {
      expect(c.label).toMatch(/^async-label-\d$/);
    }
  });

  it('works with fixture data (20 items, 4D embeddings, k=3)', async () => {
    const items = fixture.items as EmbedItem[];
    const result = await cluster(items, { k: 3, normalize: true, seed: 42 });
    expect(result.k).toBe(3);
    const total = result.clusters.reduce((s, c) => s + c.items.length, 0);
    expect(total).toBe(20);
  });

  it('fixture: silhouette score > 0.5 for well-separated clusters', async () => {
    const items = fixture.items as EmbedItem[];
    const result = await cluster(items, { k: 3, normalize: true, seed: 42 });
    expect(result.quality.silhouette.score).toBeGreaterThan(0.5);
  });
});

// ---------------------------------------------------------------------------
// findOptimalK()
// ---------------------------------------------------------------------------

describe('findOptimalK()', () => {
  it('returns an OptimalKResult with a k value', () => {
    const result = findOptimalK(ITEMS_3, { normalize: false, seed: 1 });
    expect(result.k).toBeGreaterThanOrEqual(2);
  });

  it('method is "silhouette"', () => {
    const result = findOptimalK(ITEMS_3, { normalize: false, seed: 1 });
    expect(result.method).toBe('silhouette');
  });

  it('scores array has an entry per k tried', () => {
    const result = findOptimalK(ITEMS_3, { normalize: false, seed: 1, maxK: 4 });
    // kMin=2, kMax=min(4, n-1)
    expect(result.scores.length).toBeGreaterThanOrEqual(1);
    for (const entry of result.scores) {
      expect(entry.k).toBeGreaterThanOrEqual(2);
      expect(typeof entry.silhouette).toBe('number');
      expect(typeof entry.inertia).toBe('number');
    }
  });

  it('returns k >= 3 for clearly 3-cluster data (silhouette selects good k)', () => {
    const result = findOptimalK(ITEMS_3, { normalize: false, seed: 1, maxK: 5 });
    // The selected k must be at least 3 (the true cluster count); silhouette
    // may also prefer tighter sub-clusters when maxK allows it.
    expect(result.k).toBeGreaterThanOrEqual(3);
  });

  it('selected k has the highest silhouette score in scores', () => {
    const result = findOptimalK(ITEMS_3, { normalize: false, seed: 1, maxK: 5 });
    const best = result.scores.reduce((a, b) => (a.silhouette > b.silhouette ? a : b));
    expect(result.k).toBe(best.k);
  });

  it('fixture: findOptimalK returns k=3 for 3-cluster fixture', () => {
    const items = fixture.items as EmbedItem[];
    const result = findOptimalK(items, { normalize: true, seed: 42, maxK: 6 });
    expect(result.k).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// createClusterer()
// ---------------------------------------------------------------------------

describe('createClusterer()', () => {
  it('returns an object with cluster, findOptimalK, silhouetteScore', () => {
    const c = createClusterer({ normalize: false, seed: 1 });
    expect(typeof c.cluster).toBe('function');
    expect(typeof c.findOptimalK).toBe('function');
    expect(typeof c.silhouetteScore).toBe('function');
  });

  it('cluster() uses bound config', async () => {
    const c = createClusterer({ normalize: false, seed: 1 });
    const result = await c.cluster(ITEMS_2, { k: 2 });
    expect(result.k).toBe(2);
    expect(result.clusters).toHaveLength(2);
  });

  it('cluster() options override bound config', async () => {
    const c = createClusterer({ normalize: false, seed: 1, k: 2 });
    // Override k=2 with k=1
    const result = await c.cluster(ITEMS_2, { k: 1 });
    expect(result.k).toBe(1);
  });

  it('findOptimalK() uses bound config', async () => {
    const c = createClusterer({ normalize: false, seed: 1 });
    const result = await c.findOptimalK(ITEMS_3, { maxK: 5 });
    expect(result.k).toBeGreaterThanOrEqual(2);
  });

  it('silhouetteScore() returns SilhouetteResult', async () => {
    const c = createClusterer({ normalize: false, seed: 1 });
    const result = await c.cluster(ITEMS_2, { k: 2 });
    const sil = c.silhouetteScore(result);
    expect(typeof sil.score).toBe('number');
    expect(sil.perCluster).toHaveLength(2);
  });

  it('two clusterers with different seeds can produce different assignments', async () => {
    // Use a case where seed matters: run on data where init order could vary
    const c1 = createClusterer({ normalize: false, seed: 1 });
    const c2 = createClusterer({ normalize: false, seed: 99999 });
    const r1 = await c1.cluster(ITEMS_3, { k: 3 });
    const r2 = await c2.cluster(ITEMS_3, { k: 3 });
    // Both should still produce 3 clusters — the assignments may differ
    expect(r1.clusters).toHaveLength(3);
    expect(r2.clusters).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// silhouetteScore re-export from clusterer
// ---------------------------------------------------------------------------

describe('silhouetteScore re-export', () => {
  it('silhouetteScore from clusterer matches silhouetteScore from silhouette module', async () => {
    const result = await cluster(ITEMS_2, { k: 2, normalize: false, seed: 1 });
    const s1 = silhouetteScore(result);
    const s2 = silhouetteScoreFromModule(result);
    expect(s1.score).toBeCloseTo(s2.score, 10);
  });
});

describe('findOptimalK input validation', () => {
  it('throws for empty input', () => {
    expect(() => findOptimalK([])).toThrow(ClusterError);
  });

  it('throws for fewer than 3 items', () => {
    const items: EmbedItem[] = [
      { id: '1', text: 'a', embedding: [1, 0] },
      { id: '2', text: 'b', embedding: [0, 1] },
    ];
    expect(() => findOptimalK(items)).toThrow(ClusterError);
  });
});

describe('original embeddings preserved', () => {
  it('cluster result preserves original embeddings, not normalized', async () => {
    const originalEmbedding = [...ITEMS_2[0].embedding];
    const result = await cluster(ITEMS_2, { k: 2, normalize: true, seed: 1 });
    const item = result.clusters.flatMap(c => c.items).find(i => i.id === ITEMS_2[0].id);
    expect(item).toBeDefined();
    expect(item!.embedding).toEqual(originalEmbedding);
  });
});

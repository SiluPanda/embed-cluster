import { describe, it, expect } from 'vitest';
import type {
  EmbedItem,
  ClusterItem,
  ClusterOptions,
  ClusterResult,
  Clusterer,
  Cluster,
  ClusterQuality,
  SilhouetteResult,
} from '../types';
import type { ClusterErrorCode } from '../errors';

describe('types — compile-time checks', () => {
  it('EmbedItem requires id, text, embedding', () => {
    const item: EmbedItem = {
      id: 'test-id',
      text: 'hello world',
      embedding: [0.1, 0.2, 0.3],
    };
    expect(item.id).toBe('test-id');
    expect(item.text).toBe('hello world');
    expect(item.embedding).toEqual([0.1, 0.2, 0.3]);
  });

  it('EmbedItem supports optional metadata', () => {
    const item: EmbedItem = {
      id: 'x',
      text: 'y',
      embedding: [1],
      metadata: { source: 'web', timestamp: 1234567890 },
    };
    expect(item.metadata?.source).toBe('web');
  });

  it('ClusterItem extends EmbedItem with clusterId and distanceToCentroid', () => {
    const ci: ClusterItem = {
      id: 'ci-1',
      text: 'some text',
      embedding: [0.5, 0.5],
      clusterId: 2,
      distanceToCentroid: 0.12,
    };
    expect(ci.clusterId).toBe(2);
    expect(ci.distanceToCentroid).toBe(0.12);
    // Inherited from EmbedItem
    expect(ci.id).toBe('ci-1');
    expect(ci.text).toBe('some text');
  });

  it('ClusterOptions has all-optional fields', () => {
    // An empty object should be a valid ClusterOptions
    const opts: ClusterOptions = {};
    expect(opts).toBeDefined();

    const fullOpts: ClusterOptions = {
      k: 5,
      autoK: true,
      maxK: 10,
      maxIterations: 100,
      tolerance: 1e-4,
      seed: 42,
      normalize: true,
      labeler: (_items, _id) => 'topic',
      distanceFn: (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0),
    };
    expect(fullOpts.k).toBe(5);
  });

  it('ClusterResult has clusters, quality, k, converged', () => {
    const silhouette: SilhouetteResult = { score: 0.8, perCluster: [0.75, 0.85] };
    const quality: ClusterQuality = { silhouette, inertia: 0.42 };
    const result: ClusterResult = {
      clusters: [],
      quality,
      k: 3,
      iterations: 15,
      converged: true,
      durationMs: 123,
    };
    expect(result.clusters).toEqual([]);
    expect(result.k).toBe(3);
    expect(result.converged).toBe(true);
    expect(result.quality.inertia).toBe(0.42);
  });

  it('Clusterer interface can be mock-implemented', () => {
    const mockClusterer: Clusterer = {
      cluster: async (_items, _options) => {
        const silhouette: SilhouetteResult = { score: 0.5, perCluster: [] };
        const quality: ClusterQuality = { silhouette, inertia: 0 };
        const result: ClusterResult = {
          clusters: [],
          quality,
          k: 2,
          iterations: 10,
          converged: true,
          durationMs: 50,
        };
        return result;
      },
      findOptimalK: async (_items, _options) => ({
        k: 3,
        scores: [],
        method: 'combined' as const,
      }),
      silhouetteScore: (_result) => ({ score: 0.6, perCluster: [] }),
    };

    expect(mockClusterer).toBeDefined();
    expect(typeof mockClusterer.cluster).toBe('function');
    expect(typeof mockClusterer.findOptimalK).toBe('function');
    expect(typeof mockClusterer.silhouetteScore).toBe('function');
  });

  it('ClusterErrorCode union has 5 values', () => {
    const codes: ClusterErrorCode[] = [
      'EMPTY_INPUT',
      'INCONSISTENT_DIMENSIONS',
      'DEGENERATE_INPUT',
      'INVALID_K',
      'INVALID_OPTIONS',
    ];
    expect(codes).toHaveLength(5);
    expect(codes).toContain('EMPTY_INPUT');
    expect(codes).toContain('INCONSISTENT_DIMENSIONS');
    expect(codes).toContain('DEGENERATE_INPUT');
    expect(codes).toContain('INVALID_K');
    expect(codes).toContain('INVALID_OPTIONS');
  });

  it('Cluster has required fields', () => {
    const cluster: Cluster = {
      id: 0,
      centroid: [1, 0, 0],
      items: [],
      size: 0,
      avgDistanceToCentroid: 0,
      cohesion: 0,
    };
    expect(cluster.id).toBe(0);
    expect(cluster.centroid).toEqual([1, 0, 0]);
  });
});

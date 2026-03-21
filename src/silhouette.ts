import type { ClusterResult, SilhouetteResult } from './types';
import { euclideanDistance } from './kmeans';

// ---------------------------------------------------------------------------
// Silhouette scoring
// ---------------------------------------------------------------------------

/**
 * Compute the silhouette score for a clustering result.
 *
 * Per-item silhouette coefficient:
 *   s(i) = (b(i) - a(i)) / max(a(i), b(i))
 *
 * where:
 *   a(i) = mean distance from item i to all other items in the same cluster
 *   b(i) = mean distance from item i to all items in the nearest other cluster
 *
 * Returns value in [-1, 1]; higher is better.
 * If only one cluster exists, returns 0 for all items.
 */
export function silhouetteScore(
  result: ClusterResult,
  distFn: (a: number[], b: number[]) => number = euclideanDistance,
): SilhouetteResult {
  const { clusters } = result;

  // Need at least 2 clusters to compute silhouette
  if (clusters.length < 2) {
    return {
      score: 0,
      perCluster: clusters.map(() => 0),
      perItem: clusters.flatMap(c => c.items.map(() => 0)),
    };
  }

  const allPerItem: number[] = [];
  const perCluster: number[] = [];

  for (const cluster of clusters) {
    if (cluster.items.length === 0) {
      perCluster.push(0);
      continue;
    }

    const clusterScores: number[] = [];

    for (const item of cluster.items) {
      const v = item.embedding;

      // a(i): mean intra-cluster distance
      let a = 0;
      if (cluster.items.length > 1) {
        let aSum = 0;
        for (const other of cluster.items) {
          if (other.id !== item.id) {
            aSum += distFn(v, other.embedding);
          }
        }
        a = aSum / (cluster.items.length - 1);
      }
      // If only 1 item in cluster, a = 0

      // b(i): mean distance to nearest other cluster
      let b = Infinity;
      for (const otherCluster of clusters) {
        if (otherCluster.id === cluster.id) continue;
        if (otherCluster.items.length === 0) continue;

        let bSum = 0;
        for (const other of otherCluster.items) {
          bSum += distFn(v, other.embedding);
        }
        const bMean = bSum / otherCluster.items.length;
        if (bMean < b) b = bMean;
      }

      // Silhouette for this item
      let s: number;
      if (b === Infinity) {
        // Only one non-empty cluster
        s = 0;
      } else {
        const maxAB = Math.max(a, b);
        s = maxAB === 0 ? 0 : (b - a) / maxAB;
      }

      clusterScores.push(s);
      allPerItem.push(s);
    }

    const clusterMean =
      clusterScores.length > 0
        ? clusterScores.reduce((sum, s) => sum + s, 0) / clusterScores.length
        : 0;
    perCluster.push(clusterMean);
  }

  const overallScore =
    allPerItem.length > 0
      ? allPerItem.reduce((sum, s) => sum + s, 0) / allPerItem.length
      : 0;

  return {
    score: overallScore,
    perCluster,
    perItem: allPerItem,
  };
}

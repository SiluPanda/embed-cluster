import type { EmbedItem, OptimalKResult, ClusterOptions } from './types';
import { kMeans } from './kmeans';
import { silhouetteScore } from './silhouette';

// ---------------------------------------------------------------------------
// Automatic k selection via silhouette scoring
// ---------------------------------------------------------------------------

/**
 * Try k from kMin to kMax, run k-means for each, compute silhouette score,
 * and return the k that maximises the silhouette score.
 *
 * Defaults:
 *   kMin = 2
 *   kMax = min(10, floor(sqrt(n)))
 */
export function findOptimalK(
  items: EmbedItem[],
  options: Omit<ClusterOptions, 'k'> = {},
): OptimalKResult {
  const n = items.length;
  const kMin = 2;
  const kMax = options.maxK ?? Math.min(10, Math.floor(Math.sqrt(n)));

  // Guard: need at least 2 items and kMax >= kMin
  const effectiveKMax = Math.min(kMax, n - 1);
  const effectiveKMin = Math.min(kMin, effectiveKMax);

  const scores: Array<{ k: number; silhouette: number; inertia: number }> = [];

  for (let k = effectiveKMin; k <= effectiveKMax; k++) {
    const result = kMeans(items, k, { ...options, k });
    const sil = silhouetteScore(result);
    scores.push({
      k,
      silhouette: sil.score,
      inertia: result.quality.inertia,
    });
  }

  // Select k with highest silhouette score
  let bestK = effectiveKMin;
  let bestScore = -Infinity;
  for (const entry of scores) {
    if (entry.silhouette > bestScore) {
      bestScore = entry.silhouette;
      bestK = entry.k;
    }
  }

  return {
    k: bestK,
    scores,
    method: 'silhouette',
  };
}

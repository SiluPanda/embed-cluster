import type { EmbedItem, ClusterResult, ClusterOptions, OptimalKResult, Clusterer } from './types';
import { ClusterError } from './errors';
import { kMeans } from './kmeans';
import { silhouetteScore } from './silhouette';
import { findOptimalK } from './optimal-k';

export { findOptimalK } from './optimal-k';
export { silhouetteScore } from './silhouette';

// ---------------------------------------------------------------------------
// Convenience top-level cluster() function
// ---------------------------------------------------------------------------

/**
 * Cluster a set of EmbedItems using k-means++ and return a ClusterResult
 * with silhouette scores populated in quality.silhouette.
 *
 * Provide either `options.k` (fixed) or `options.autoK = true` to
 * auto-select the optimal k.
 */
export async function cluster(
  items: EmbedItem[],
  options: ClusterOptions = {},
): Promise<ClusterResult> {
  if (items.length === 0) {
    throw new ClusterError('Input must not be empty', 'EMPTY_INPUT');
  }

  let k: number;
  if (options.autoK) {
    const optResult = findOptimalK(items, options);
    k = optResult.k;
  } else if (options.k !== undefined) {
    k = options.k;
  } else {
    throw new ClusterError(
      'Provide options.k or set options.autoK = true',
      'INVALID_OPTIONS',
    );
  }

  const result = kMeans(items, k, options);

  // Populate silhouette scores in quality
  const sil = silhouetteScore(result);
  result.quality.silhouette = sil;

  // Apply labeler if provided
  if (options.labeler) {
    for (const c of result.clusters) {
      const label = await Promise.resolve(options.labeler(c.items, c.id));
      c.label = label;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// createClusterer factory
// ---------------------------------------------------------------------------

/**
 * Create a pre-configured Clusterer instance bound to the given config.
 * The returned object exposes cluster(), findOptimalK(), and silhouetteScore().
 */
export function createClusterer(config: ClusterOptions = {}): Clusterer {
  return {
    async cluster(items: EmbedItem[], options: ClusterOptions = {}): Promise<ClusterResult> {
      return cluster(items, { ...config, ...options });
    },

    async findOptimalK(
      items: EmbedItem[],
      options: Omit<ClusterOptions, 'k'> = {},
    ): Promise<OptimalKResult> {
      return findOptimalK(items, { ...config, ...options });
    },

    silhouetteScore(result: ClusterResult) {
      return silhouetteScore(result);
    },
  };
}

export type {
  EmbedItem, ClusterItem, Cluster, SilhouetteResult, OptimalKResult,
  ClusterQuality, VisualizationData, LabelerFn,
  ClusterOptions, ClusterResult, Clusterer,
} from './types';
export { ClusterError } from './errors';
export type { ClusterErrorCode } from './errors';
export { normalizeVector, normalizeVectors } from './normalize';
// cluster, findOptimalK, silhouetteScore, createClusterer — Phase 2+

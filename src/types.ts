export interface EmbedItem {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

export interface ClusterItem extends EmbedItem {
  clusterId: number;
  distanceToCentroid: number;
}

export interface Cluster {
  id: number;
  centroid: number[];
  items: ClusterItem[];
  label?: string;
  size: number;
  avgDistanceToCentroid: number;
  cohesion: number;       // average intra-cluster distance
}

export interface SilhouetteResult {
  score: number;          // overall mean silhouette coefficient (-1 to 1)
  perCluster: number[];   // per-cluster mean silhouette
  perItem?: number[];     // per-item silhouette (optional, expensive)
}

export interface OptimalKResult {
  k: number;
  scores: Array<{ k: number; silhouette: number; inertia: number }>;
  method: 'silhouette' | 'elbow' | 'combined';
}

export interface ClusterQuality {
  silhouette: SilhouetteResult;
  inertia: number;         // sum of squared distances to centroids
  daviesBouldin?: number;  // optional DB index
  calinski?: number;       // optional Calinski-Harabasz index
}

export interface VisualizationData {
  points: Array<{ id: string; x: number; y: number; clusterId: number }>;
  method: 'pca' | 'umap' | 'tsne';
}

export type LabelerFn = (
  items: EmbedItem[],
  clusterId: number,
) => Promise<string> | string;

export interface ClusterOptions {
  k?: number;                    // number of clusters (required if autoK is false)
  autoK?: boolean;               // auto-select k (default false)
  maxK?: number;                 // max k to try when autoK=true (default 10)
  maxIterations?: number;        // k-means max iterations (default 100)
  tolerance?: number;            // convergence tolerance (default 1e-4)
  seed?: number;                 // random seed for reproducibility
  normalize?: boolean;           // L2-normalize embeddings before clustering (default true)
  labeler?: LabelerFn;           // auto-labeling function
  distanceFn?: (a: number[], b: number[]) => number;  // custom distance
}

export interface ClusterResult {
  clusters: Cluster[];
  quality: ClusterQuality;
  k: number;
  iterations: number;
  converged: boolean;
  durationMs: number;
}

export interface Clusterer {
  cluster(items: EmbedItem[], options?: ClusterOptions): Promise<ClusterResult>;
  findOptimalK(items: EmbedItem[], options?: Omit<ClusterOptions, 'k'>): Promise<OptimalKResult>;
  silhouetteScore(result: ClusterResult): SilhouetteResult;
}

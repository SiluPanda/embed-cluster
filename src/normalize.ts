/**
 * L2-normalize a single vector. Returns a unit-length copy.
 * If the vector has zero magnitude, returns a zero vector (unchanged).
 */
export function normalizeVector(vec: number[]): number[] {
  const magnitude = Math.sqrt(vec.reduce((sum, x) => sum + x * x, 0));
  if (magnitude === 0) return [...vec];
  return vec.map(x => x / magnitude);
}

/**
 * L2-normalize a batch of vectors. Each vector is normalized independently.
 */
export function normalizeVectors(vecs: number[][]): number[][] {
  return vecs.map(normalizeVector);
}

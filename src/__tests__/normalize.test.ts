import { describe, it, expect } from 'vitest';
import { normalizeVector, normalizeVectors } from '../normalize';

const EPSILON = 1e-10;

describe('normalizeVector', () => {
  it('normalizes [3, 4] to [0.6, 0.8] (magnitude=5)', () => {
    const result = normalizeVector([3, 4]);
    expect(result[0]).toBeCloseTo(0.6, 10);
    expect(result[1]).toBeCloseTo(0.8, 10);
  });

  it('normalizes a unit vector [1, 0, 0] unchanged', () => {
    const result = normalizeVector([1, 0, 0]);
    expect(result[0]).toBeCloseTo(1.0, 10);
    expect(result[1]).toBeCloseTo(0.0, 10);
    expect(result[2]).toBeCloseTo(0.0, 10);
  });

  it('returns [0, 0, 0] for zero vector', () => {
    const result = normalizeVector([0, 0, 0]);
    expect(result).toEqual([0, 0, 0]);
  });

  it('returns a copy and does not mutate input', () => {
    const input = [3, 4];
    const result = normalizeVector(input);
    expect(result).not.toBe(input);
    expect(input[0]).toBe(3);
    expect(input[1]).toBe(4);
  });

  it('normalizes [2, 2] to approximately [0.707, 0.707]', () => {
    const result = normalizeVector([2, 2]);
    const expected = Math.SQRT2 / 2;
    expect(result[0]).toBeCloseTo(expected, 10);
    expect(result[1]).toBeCloseTo(expected, 10);
  });

  it('resulting magnitude is approximately 1.0', () => {
    const vecs = [
      [3, 4],
      [1, 2, 3],
      [0.5, -0.3, 0.8, 0.1],
      [100, 200, 300],
    ];
    for (const vec of vecs) {
      const norm = normalizeVector(vec);
      const magnitude = Math.sqrt(norm.reduce((s, x) => s + x * x, 0));
      expect(Math.abs(magnitude - 1.0)).toBeLessThan(EPSILON);
    }
  });

  it('handles single-element vector', () => {
    const result = normalizeVector([5]);
    expect(result[0]).toBeCloseTo(1.0, 10);
  });

  it('handles negative values', () => {
    const result = normalizeVector([-3, 4]);
    expect(result[0]).toBeCloseTo(-0.6, 10);
    expect(result[1]).toBeCloseTo(0.8, 10);
    const magnitude = Math.sqrt(result.reduce((s, x) => s + x * x, 0));
    expect(Math.abs(magnitude - 1.0)).toBeLessThan(EPSILON);
  });
});

describe('normalizeVectors', () => {
  it('normalizes [[3,4],[0,1]] correctly', () => {
    const result = normalizeVectors([[3, 4], [0, 1]]);
    expect(result[0][0]).toBeCloseTo(0.6, 10);
    expect(result[0][1]).toBeCloseTo(0.8, 10);
    expect(result[1][0]).toBeCloseTo(0.0, 10);
    expect(result[1][1]).toBeCloseTo(1.0, 10);
  });

  it('returns empty array for empty input', () => {
    const result = normalizeVectors([]);
    expect(result).toEqual([]);
  });

  it('returns array of same length as input', () => {
    const input = [[1, 0], [0, 1], [1, 1]];
    const result = normalizeVectors(input);
    expect(result).toHaveLength(3);
  });

  it('each normalized vector has magnitude approximately 1.0', () => {
    const input = [[1, 2], [3, 4], [5, 12]];
    const result = normalizeVectors(input);
    for (const vec of result) {
      const magnitude = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
      expect(Math.abs(magnitude - 1.0)).toBeLessThan(EPSILON);
    }
  });

  it('handles zero vectors in batch without throwing', () => {
    const result = normalizeVectors([[0, 0], [1, 0]]);
    expect(result[0]).toEqual([0, 0]);
    expect(result[1][0]).toBeCloseTo(1.0, 10);
    expect(result[1][1]).toBeCloseTo(0.0, 10);
  });
});

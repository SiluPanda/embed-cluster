import { describe, it, expect } from 'vitest';
import fixture from './fixtures/embeddings-small.json';

describe('embeddings-small.json fixture', () => {
  it('loads and parses', () => {
    expect(fixture).toBeDefined();
    expect(typeof fixture).toBe('object');
  });

  it('has exactly 20 items', () => {
    expect(fixture.items).toHaveLength(20);
  });

  it('each item has id, text, embedding fields', () => {
    for (const item of fixture.items) {
      expect(typeof item.id).toBe('string');
      expect(typeof item.text).toBe('string');
      expect(Array.isArray(item.embedding)).toBe(true);
    }
  });

  it('all embeddings have length 4', () => {
    for (const item of fixture.items) {
      expect(item.embedding).toHaveLength(4);
    }
  });

  it('all embedding values are numbers', () => {
    for (const item of fixture.items) {
      for (const val of item.embedding) {
        expect(typeof val).toBe('number');
      }
    }
  });

  it('all item ids are unique', () => {
    const ids = fixture.items.map((i) => i.id);
    const unique = new Set(ids);
    expect(unique.size).toBe(20);
  });

  it('has a description field', () => {
    expect(typeof fixture.description).toBe('string');
  });
});

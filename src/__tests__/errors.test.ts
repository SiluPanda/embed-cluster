import { describe, it, expect } from 'vitest';
import { ClusterError } from '../errors';
import type { ClusterErrorCode } from '../errors';

describe('ClusterError', () => {
  it('extends Error', () => {
    const err = new ClusterError('something failed', 'EMPTY_INPUT');
    expect(err).toBeInstanceOf(Error);
  });

  it('name is ClusterError', () => {
    const err = new ClusterError('msg', 'INVALID_K');
    expect(err.name).toBe('ClusterError');
  });

  it('code is accessible', () => {
    const err = new ClusterError('bad k', 'INVALID_K');
    expect(err.code).toBe('INVALID_K');
  });

  it('instanceof ClusterError is correct', () => {
    const err = new ClusterError('test', 'INVALID_OPTIONS');
    expect(err).toBeInstanceOf(ClusterError);
  });

  it('message is preserved', () => {
    const msg = 'embeddings array is empty';
    const err = new ClusterError(msg, 'EMPTY_INPUT');
    expect(err.message).toBe(msg);
  });

  it('constructs with EMPTY_INPUT code', () => {
    const err = new ClusterError('no items', 'EMPTY_INPUT');
    expect(err.code).toBe('EMPTY_INPUT');
  });

  it('constructs with INCONSISTENT_DIMENSIONS code', () => {
    const err = new ClusterError('dim mismatch', 'INCONSISTENT_DIMENSIONS');
    expect(err.code).toBe('INCONSISTENT_DIMENSIONS');
  });

  it('constructs with DEGENERATE_INPUT code', () => {
    const err = new ClusterError('all zeros', 'DEGENERATE_INPUT');
    expect(err.code).toBe('DEGENERATE_INPUT');
  });

  it('constructs with INVALID_K code', () => {
    const err = new ClusterError('k must be >= 2', 'INVALID_K');
    expect(err.code).toBe('INVALID_K');
  });

  it('constructs with INVALID_OPTIONS code', () => {
    const err = new ClusterError('bad options', 'INVALID_OPTIONS');
    expect(err.code).toBe('INVALID_OPTIONS');
  });

  it('all 5 error codes construct correctly', () => {
    const codes: ClusterErrorCode[] = [
      'EMPTY_INPUT',
      'INCONSISTENT_DIMENSIONS',
      'DEGENERATE_INPUT',
      'INVALID_K',
      'INVALID_OPTIONS',
    ];
    for (const code of codes) {
      const err = new ClusterError(`error: ${code}`, code);
      expect(err).toBeInstanceOf(ClusterError);
      expect(err).toBeInstanceOf(Error);
      expect(err.code).toBe(code);
      expect(err.name).toBe('ClusterError');
    }
  });

  it('stack trace is populated', () => {
    const err = new ClusterError('trace check', 'EMPTY_INPUT');
    expect(err.stack).toBeDefined();
  });
});

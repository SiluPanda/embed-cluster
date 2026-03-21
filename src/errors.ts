export type ClusterErrorCode =
  | 'EMPTY_INPUT'
  | 'INCONSISTENT_DIMENSIONS'
  | 'DEGENERATE_INPUT'
  | 'INVALID_K'
  | 'INVALID_OPTIONS';

export class ClusterError extends Error {
  readonly name = 'ClusterError';
  constructor(
    message: string,
    readonly code: ClusterErrorCode,
  ) {
    super(message);
    Object.setPrototypeOf(this, ClusterError.prototype);
  }
}

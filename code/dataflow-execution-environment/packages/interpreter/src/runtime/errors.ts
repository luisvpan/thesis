export type ErrorCode =
  | "PARSE_ERROR"
  | "DIVISION_BY_ZERO"
  | "CIRCULAR_DEPENDENCY"
  | "EVALUATION_CYCLE"
  | "UNDEFINED_REFERENCE"
  | "TYPE_ERROR"
  | "UNKNOWN_OPERATION"
  | "INVALID_ARGUMENT"
  | "ARITY_ERROR";

export class RuntimeError extends Error {
  constructor(
    public code: ErrorCode,
    message: string,
    public nodeId?: string
  ) {
    super(`[${code}]${nodeId ? ` at '${nodeId}'` : ""}: ${message}`);
    this.name = "RuntimeError";
  }
}

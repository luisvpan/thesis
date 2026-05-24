export type ErrorCode =
  | "PARSE_ERROR"
  | "DIVISION_BY_ZERO"
  | "CIRCULAR_DEPENDENCY"
  | "EVALUATION_CYCLE"
  | "UNDEFINED_REFERENCE"
  | "TYPE_ERROR"
  | "UNKNOWN_OPERATION"
  | "INVALID_ARGUMENT"
  | "INVALID_OBJECT"
  | "ARITY_ERROR";

export class RuntimeError extends Error {
  code: ErrorCode;
  nodeId?: string;

  constructor(code: ErrorCode, message: string, nodeId?: string) {
    super(`[${code}]${nodeId ? ` at '${nodeId}'` : ""}: ${message}`);
    this.name = "RuntimeError";
    this.code = code;
    this.nodeId = nodeId;
  }
}

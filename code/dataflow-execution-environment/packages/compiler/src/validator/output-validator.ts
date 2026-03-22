import type { ValidationError } from "@dataflow/shared/types";
import type { Statement } from "../ast/ast-types.js";
import { ERROR_MESSAGES } from "./messages.js";

export function checkOutputNode(statements: Statement[]): ValidationError | null {
  const hasOutputNode = statements.some(stmt => stmt.type === "OutputStatement");
  if (!hasOutputNode) {
    return {
      code: "MISSING_OUTPUT_NODE",
      message: "Program must have at least one output node",
      childMessage: ERROR_MESSAGES.MISSING_OUTPUT_NODE.childMessage(),
      suggestion: ERROR_MESSAGES.MISSING_OUTPUT_NODE.suggestion(),
      example: ERROR_MESSAGES.MISSING_OUTPUT_NODE.example()
    };
  }
  return null;
}

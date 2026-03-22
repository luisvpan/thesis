import type { ValidationError } from "@dataflow/shared/types";
import { OPERATION_REGISTRY } from "@dataflow/shared/operations";
import type { Statement } from "../ast/ast-types.js";
import { ERROR_MESSAGES } from "./messages.js";

export function checkArity(stmt: Statement): ValidationError | null {
  if (stmt.type !== "TransformStatement") {
    return null;
  }

  const signatures = OPERATION_REGISTRY[stmt.operation];
  if (!signatures) {
    return null;
  }

  const arities = new Set(signatures.contracts.map(c => c.arity));
  if (!arities.has(stmt.inputs.length)) {
    const expectedArity = Array.from(arities).sort((a, b) => a - b)[0];
    return {
      code: "WRONG_ARITY",
      message: `Operation ${stmt.operation} requires ${expectedArity} ${expectedArity === 1 ? 'input' : 'inputs'}, got ${stmt.inputs.length}`,
      nodeId: stmt.id,
      childMessage: ERROR_MESSAGES.WRONG_ARITY.childMessage(stmt.operation, expectedArity, stmt.inputs.length),
      suggestion: ERROR_MESSAGES.WRONG_ARITY.suggestion(stmt.operation, expectedArity, stmt.inputs.length),
      example: ERROR_MESSAGES.WRONG_ARITY.example(stmt.operation, expectedArity)
    };
  }

  return null;
}

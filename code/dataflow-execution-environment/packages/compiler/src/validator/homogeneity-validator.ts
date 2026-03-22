import type { ValidationError } from "@dataflow/shared/types";
import type { Statement } from "../ast/ast-types.js";
import { ERROR_MESSAGES } from "./messages.js";
import { inferType } from "./type-checker.js";

export function validateSetHomogeneity(stmt: Statement): ValidationError[] {
  if (stmt.type !== "SourceStatement") {
    return [];
  }

  const dataType = stmt.dataType as string;
  if (!dataType.startsWith("set<")) {
    return [];
  }

  const value = stmt.value as unknown[];
  if (!Array.isArray(value) || value.length === 0) {
    return [];
  }

  const setMatch = dataType.match(/^set<(.+)>$/);
  if (!setMatch) {
    return [];
  }

  const declaredElementType = setMatch[1];

  for (const element of value) {
    const elementType = inferType(element, declaredElementType);

    if (elementType && elementType !== declaredElementType) {
      return [{
        code: "SET_HETEROGENEITY_ERROR",
        message: "All elements in set must have same type",
        nodeId: stmt.id,
        childMessage: ERROR_MESSAGES.SET_HETEROGENEITY_ERROR.childMessage(),
        suggestion: ERROR_MESSAGES.SET_HETEROGENEITY_ERROR.suggestion(declaredElementType),
        example: ERROR_MESSAGES.SET_HETEROGENEITY_ERROR.example(declaredElementType)
      }];
    }
  }

  return [];
}

export function validateLiteralType(stmt: Statement): ValidationError[] {
  if (stmt.type !== "SourceStatement") {
    return [];
  }

  if (!stmt.value) {
    return [];
  }

  const declaredType = stmt.dataType as string;

  const setMatch = declaredType.match(/^set<(.+)>$/);
  if (setMatch) {
    const elementType = setMatch[1];
    const valueArray = stmt.value as unknown[];

    if (Array.isArray(valueArray)) {
      for (let i = 0; i < valueArray.length; i++) {
        const elementValue = valueArray[i];
        const valueType = inferType(elementValue, elementType);

        if (valueType && valueType !== elementType) {
          return [{
            code: "LITERAL_TYPE_MISMATCH",
            message: `Set element at index ${i} has type ${valueType}, but set expects ${elementType}`,
            nodeId: stmt.id,
            childMessage: ERROR_MESSAGES.LITERAL_TYPE_MISMATCH.childMessage(i, elementType),
            suggestion: ERROR_MESSAGES.LITERAL_TYPE_MISMATCH.suggestion(elementType),
            example: ERROR_MESSAGES.LITERAL_TYPE_MISMATCH.example(elementType)
          }];
        }
      }
    }
    return [];
  }

  const valueType = inferType(stmt.value);
  if (valueType && valueType !== declaredType) {
    return [{
      code: "LITERAL_TYPE_MISMATCH",
      message: `Literal value ${JSON.stringify(stmt.value)} has type ${valueType}, but declared type is ${declaredType}`,
      nodeId: stmt.id,
      childMessage: "⚠️ ¡Ups! El valor no coincide con el tipo.",
      suggestion: ERROR_MESSAGES.LITERAL_TYPE_MISMATCH.suggestion(declaredType),
      example: ERROR_MESSAGES.LITERAL_TYPE_MISMATCH.example(declaredType)
    }];
  }

  return [];
}

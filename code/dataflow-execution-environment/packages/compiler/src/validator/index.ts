import type { DataflowNode, DataValue, DataType, TypeExpression } from "@dataflow/shared/types";
import type { ValidationError, ValidationResult } from "@dataflow/shared/types";
import type { Statement } from "../ast/ast-types.js";
import { OPERATION_REGISTRY, TypeConstraint, resolveOperationSignature } from "@dataflow/shared/operations";
import type { DataflowProgram } from "@dataflow/shared/types";
import { ERROR_MESSAGES } from "./messages.js";
import { checkArity } from "./arity-validator.js";
import { detectCycles } from "./dag-validator.js";
import { validateSetHomogeneity, validateLiteralType } from "./homogeneity-validator.js";
import { checkOutputNode } from "./output-validator.js";
import { 
  inferType, 
  stringToTypeExpression, 
  stringToDataType,
  getInputType,
  isTypeCompatible,
  extractElementType,
  validatePropertyConstraints,
  typeHasProperty
} from "./type-checker.js";

export class DagValidator {
  private requireOutputNode: boolean;

  constructor(requireOutputNode: boolean = false) {
    this.requireOutputNode = requireOutputNode;
  }

  validateProgram(program: DataflowProgram, requireOutputNode?: boolean): ValidationResult {
    const statements: Statement[] = program.graph.nodes.map(node => {
      if (node.type === "DataSource") {
        return {
          type: "SourceStatement",
          id: node.id,
          dataType: node.dataType as string,
          value: node.value
        };
      } else if (node.type === "Transformation") {
        return {
          type: "TransformStatement",
          id: node.id,
          dataType: node.dataType as string,
          operation: node.operation,
          inputs: node.inputs
        };
      } else {
        return {
          type: "OutputStatement",
          id: node.id,
          dataType: node.dataType as string,
          input: (node as any).input
        };
      }
    });

    return this.validate(statements, requireOutputNode ?? this.requireOutputNode);
  }

  validate(statements: Statement[], requireOutputNode: boolean = false): ValidationResult {
    const errors: ValidationError[] = [];
    const defined = new Set<string>();
    const typeTable = new Map<string, string>();

    for (const stmt of statements) {
      if (defined.has(stmt.id)) {
        errors.push({
          code: "DUPLICATE_IDENTIFIER",
          message: `Identifier '${stmt.id}' already defined`,
          nodeId: stmt.id,
          childMessage: ERROR_MESSAGES.DUPLICATE_IDENTIFIER.childMessage(stmt.id),
          suggestion: ERROR_MESSAGES.DUPLICATE_IDENTIFIER.suggestion(stmt.id),
          example: ERROR_MESSAGES.DUPLICATE_IDENTIFIER.example(stmt.id)
        });
      }
      defined.add(stmt.id);

      if (stmt.type === "TransformStatement") {
        typeTable.set(stmt.id, stmt.dataType);

        for (const input of stmt.inputs) {
          if (input.startsWith("literal_")) {
            defined.add(input);
            const match = input.match(/^literal_(\w+)_(.+)$/);
            if (match) {
              const type = match[1];
              const valueStr = match[2];
              let dataType: string;
              if (type === "number") {
                const value = JSON.parse(atob(valueStr));
                if (Number.isInteger(value)) {
                  dataType = value < 0 ? "integer" : "natural";
                } else {
                  dataType = "decimal";
                }
              } else if (type === "string") {
                dataType = "text";
              } else if (type === "boolean") {
                dataType = "boolean";
              } else if (type === "object") {
                const value = JSON.parse(atob(valueStr));
                if (typeof value === 'object' && value !== null && 'kind' in value) {
                  const kind = (value as { kind: string }).kind;
                  if (['fraction', 'shape', 'car', 'food', 'animal', 'person'].includes(kind)) {
                    dataType = kind;
                  } else {
                    dataType = "unknown";
                  }
                } else {
                  dataType = "unknown";
                }
              } else {
                dataType = "unknown";
              }
              typeTable.set(input, dataType);
            }
          }
        }
      }
    }

    for (const stmt of statements) {
      if (stmt.type === "SourceStatement") {
        typeTable.set(stmt.id, stmt.dataType);

        const setHomogeneityErrors = validateSetHomogeneity(stmt);
        errors.push(...setHomogeneityErrors);

        const literalTypeErrors = validateLiteralType(stmt);
        errors.push(...literalTypeErrors);
      }

      if (stmt.type === "TransformStatement" || stmt.type === "OutputStatement") {
        const refs = stmt.type === "TransformStatement" ? stmt.inputs : [stmt.input];
        for (const ref of refs) {
          if (!defined.has(ref)) {
            errors.push({
              code: "UNDEFINED_IDENTIFIER",
              message: `Undefined identifier: ${ref}`,
              nodeId: stmt.id,
              childMessage: ERROR_MESSAGES.UNDEFINED_IDENTIFIER.childMessage(ref),
              suggestion: ERROR_MESSAGES.UNDEFINED_IDENTIFIER.suggestion(),
              example: ERROR_MESSAGES.UNDEFINED_IDENTIFIER.example()
            });
          }
        }
      }
    }

    const hasCycle = detectCycles(statements);
    if (hasCycle) {
      errors.push({
        code: "CYCLE_DETECTED",
        message: "Graph contains a cycle",
        childMessage: ERROR_MESSAGES.CYCLE_DETECTED.childMessage(),
        suggestion: ERROR_MESSAGES.CYCLE_DETECTED.suggestion(),
        example: ERROR_MESSAGES.CYCLE_DETECTED.example()
      });
    }

    for (const stmt of statements) {
      if (stmt.type === "TransformStatement") {
        const signatures = OPERATION_REGISTRY[stmt.operation];
        if (!signatures) {
          errors.push({
            code: "UNKNOWN_OPERATION",
            message: `Unknown operation: ${stmt.operation}`,
            nodeId: stmt.id,
            childMessage: ERROR_MESSAGES.UNKNOWN_OPERATION.childMessage(stmt.operation),
            suggestion: ERROR_MESSAGES.UNKNOWN_OPERATION.suggestion(),
            example: ERROR_MESSAGES.UNKNOWN_OPERATION.example()
          });
          continue;
        }

        const inputTypes = stmt.inputs.map(input => getInputType(input, typeTable));

        const arityError = checkArity(stmt);
        if (arityError) {
          errors.push(arityError);
          continue;
        }

        const inputDataTypes = inputTypes.map(t => stringToTypeExpression(t)).filter((t): t is TypeExpression => t !== null) as TypeExpression[];

        const contract = resolveOperationSignature(stmt.operation, inputDataTypes);

        if (!contract) {
          const numericOps = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"];
          const hasNumericInput = inputTypes.some(t => ["natural", "integer", "decimal", "fraction"].includes(t));
          const hasTextInput = inputTypes.includes("text");

          let childMessage = ERROR_MESSAGES.TYPE_ERROR.childMessage(stmt.operation, inputTypes);
          let suggestion = ERROR_MESSAGES.TYPE_ERROR.suggestion(stmt.operation, inputTypes);
          let example = ERROR_MESSAGES.TYPE_ERROR.example(stmt.operation, inputTypes, signatures);

          if (stmt.operation === "SORT") {
            childMessage = `⚠️ ¡Ups! SORT solo funciona con números (natural, integer, decimal, fraction).\nTu conjunto contiene "${inputTypes.join(", ")}".`;
            suggestion = `Conecta bloques de números al bloque SORT. Para ordenar palabras, usa ALPHABETICAL_SORT.`;
            example = `[números 1, 5, 3] → SORT → [1, 3, 5] ✅`;
          } else if (stmt.operation === "ALPHABETICAL_SORT") {
            childMessage = `⚠️ ¡Ups! ALPHABETICAL_SORT solo funciona con texto (text).\nTu conjunto contiene "${inputTypes.join(", ")}".`;
            suggestion = `Conecta bloques de texto al bloque ALPHABETICAL_SORT. Para ordenar números, usa SORT.`;
            example = `[texto "z", "a", "m"] → ALPHABETICAL_SORT → ["a", "m", "z"] ✅`;
          } else if (numericOps.includes(stmt.operation) && hasNumericInput && hasTextInput) {
            childMessage = `⚠️ ¡Ups! El bloque "${stmt.operation}" necesita números, no palabras.`;
            suggestion = `Conecta bloques de números (natural, integer, decimal, fraction) al bloque "${stmt.operation}".`;
            example = `[número 5] + [número 3] ✅`;
          } else if (stmt.operation.startsWith("FILTER_BY_") || stmt.operation.startsWith("COMPARE_BY_")) {
            const property = stmt.operation.split("_BY_")[1].toLowerCase();
            childMessage = `⚠️ ¡Ups! El bloque "${stmt.operation}" necesita elementos con propiedad '${property}'.`;
            suggestion = `Usa un tipo que tenga propiedad '${property}'.`;
            example = `set<tipo con ${property}> → filtrados ✅`;
          }

          errors.push({
            code: "TYPE_ERROR",
            message: stmt.operation.startsWith("FILTER_BY_") || stmt.operation.startsWith("COMPARE_BY_")
              ? `No matching signature for operation ${stmt.operation} with input types [${inputTypes.join(", ")}] - missing '${stmt.operation.split("_BY_")[1].toLowerCase()}' property`
              : `No matching signature for operation ${stmt.operation} with input types [${inputTypes.join(", ")}]`,
            nodeId: stmt.id,
            childMessage,
            suggestion,
            example
          });

          continue;
        }

        for (let i = 0; i < stmt.inputs.length; i++) {
          const inputType = inputTypes[i];
          const expectedType = contract.inputTypes[i];

          if (inputType === "unknown") {
            continue;
          }

          const inputDataType = stringToDataType(inputType);
          if (!inputDataType) {
            continue;
          }

          if (!isTypeCompatible(inputType, expectedType, stmt.operation, i, typeHasProperty)) {
            const errorDetails = this.generateTypeError(
              stmt.operation,
              expectedType,
              inputType,
              stmt.inputs[i],
              i,
              stmt.id
            );
            errors.push(errorDetails);
          }
        }
      }
    }

    if (requireOutputNode) {
      const outputNodeError = checkOutputNode(statements);
      if (outputNodeError) {
        errors.push(outputNodeError);
      }
    }

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    };
  }

  private generateTypeError(
    operation: string,
    expectedType: TypeExpression | TypeConstraint,
    actualType: string,
    inputId: string,
    inputIndex: number,
    transformNodeId: string
  ): ValidationError {
    const expectedTypeName = typeof expectedType === "string" ? expectedType : expectedType.kind;

    if (operation.startsWith("FILTER_BY_") || operation.startsWith("COMPARE_BY_")) {
      const property = operation.split("_BY_")[1].toLowerCase();
      const expectedTypeWithProperty = typeof expectedType === "object" ? Object.keys(expectedType).includes("elementType") ? `${expectedType.kind}<${(expectedType as any).elementType}>` : `${expectedType.kind}<${(expectedType as TypeConstraint).property}>` : expectedType;
      const elementType = extractElementType(actualType);

      return {
        code: "TYPE_ERROR",
        message: `Operation ${operation} requires elements with '${property}' property, got ${actualType}`,
        nodeId: transformNodeId,
        childMessage: ERROR_MESSAGES.FILTER_COMPARE_PROPERTY_ERROR.childMessage(operation, property, elementType),
        suggestion: ERROR_MESSAGES.FILTER_COMPARE_PROPERTY_ERROR.suggestion(property, expectedTypeWithProperty),
        example: ERROR_MESSAGES.FILTER_COMPARE_PROPERTY_ERROR.example(expectedTypeWithProperty, property)
      };
    }

    return {
      code: "TYPE_ERROR",
      message: `Operation ${operation} expects ${expectedTypeName}, received ${actualType}`,
      nodeId: transformNodeId,
      childMessage: ERROR_MESSAGES.TYPE_ERROR_INPUT.childMessage(operation, actualType, expectedTypeName, inputId),
      suggestion: ERROR_MESSAGES.TYPE_ERROR_INPUT.suggestion(operation, actualType, expectedTypeName, inputId),
      example: ERROR_MESSAGES.TYPE_ERROR_INPUT.example(operation, actualType, expectedTypeName, inputId)
    };
  }
}

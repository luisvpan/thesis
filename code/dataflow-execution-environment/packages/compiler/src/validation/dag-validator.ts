import type { DataflowNode, DataflowEdge, DataType } from "@dataflow/shared/types";
import type { ValidationError, ValidationResult } from "@dataflow/shared/types";
import type { Statement } from "../ast/ast-types.js";
import { OPERATION_REGISTRY, TypeConstraint } from "@dataflow/shared/operations";
import type { Curriculum } from "@dataflow/shared/types";

export class DagValidator {
  validate(statements: Statement[]): ValidationResult {
    const errors: ValidationError[] = [];
    const defined = new Set<string>();
    const typeTable = new Map<string, string>();

    for (const stmt of statements) {
      if (defined.has(stmt.id)) {
        errors.push({
          code: "DUPLICATE_IDENTIFIER",
          message: `Identifier '${stmt.id}' already defined`,
          nodeId: stmt.id,
          childMessage: `⚠️ ¡Ups! Ya tienes un bloque llamado "${stmt.id}". Cada bloque necesita un nombre único.`,
          suggestion: `Cambia el nombre de uno de los bloques "${stmt.id}"`,
          example: `[${stmt.id}_1] y [${stmt.id}_2] ✅`
        });
      }
      defined.add(stmt.id);

      if (stmt.type === "SourceStatement") {
        typeTable.set(stmt.id, stmt.dataType);
      }
    }

    const hasCycle = this.detectCycles(statements);
    if (hasCycle) {
      errors.push({
        code: "CYCLE_DETECTED",
        message: "Graph contains a cycle",
        childMessage: "⚠️ ¡Ups! Hay un ciclo en el programa. Los bloques se conectan en círculo y no se puede calcular.",
        suggestion: "Busca dónde un bloque apunta a sí mismo y desconecta una línea.",
        example: "[A] → [B] → [C] ❌\n[A] → [B] ✅ (rompe el ciclo)"
      });
    }

    for (const stmt of statements) {
      if (stmt.type === "TransformStatement") {
        const signature = OPERATION_REGISTRY[stmt.operation];
        if (!signature) {
          errors.push({
            code: "UNKNOWN_OPERATION",
            message: `Unknown operation: ${stmt.operation}`,
            nodeId: stmt.id,
            childMessage: `⚠️ ¡Ups! No reconozco el bloque "${stmt.operation}".`,
            suggestion: `Revisa si el bloque está conectado correctamente.`,
            example: `Usa un bloque conocido como "+", "-", "×", "÷"`
          });
          continue;
        }

        if (stmt.inputs.length !== signature.arity) {
          errors.push({
            code: "WRONG_ARITY",
            message: `Operation ${stmt.operation} requires ${signature.arity} inputs, got ${stmt.inputs.length}`,
            nodeId: stmt.id,
            childMessage: `⚠️ ¡Ups! El bloque "${stmt.operation}" necesita ${signature.arity} ${signature.arity === 1 ? 'entrada' : 'entradas'}. Solo conectaste ${stmt.inputs.length}.`,
            suggestion: `Conecta ${signature.arity - stmt.inputs.length} bloque${signature.arity - stmt.inputs.length === 1 ? '' : 's'} más al bloque "${stmt.operation}".`,
            example: this.generateExample(stmt.operation, signature.arity)
          });
        }

        for (let i = 0; i < stmt.inputs.length; i++) {
          const inputType = this.getInputType(stmt.inputs[i], typeTable);
          const expectedType = signature.inputTypes[i];

          if (inputType === "unknown") {
            continue;
          }

          if (!this.isTypeCompatible(inputType, expectedType, stmt.operation, i)) {
            const errorDetails = this.generateTypeError(
              stmt.operation,
              expectedType,
              inputType,
              stmt.inputs[i],
              i
            );
            errors.push(errorDetails);
          }
        }
      }

      if (stmt.type === "TransformStatement" || stmt.type === "OutputStatement") {
        const refs = stmt.type === "TransformStatement" ? stmt.inputs : [stmt.input];
        for (const ref of refs) {
          if (!defined.has(ref)) {
            errors.push({
              code: "UNDEFINED_IDENTIFIER",
              message: `Undefined identifier: ${ref}`,
              nodeId: stmt.id,
              childMessage: `⚠️ ¡Ups! No encuentro el bloque "${ref}".`,
              suggestion: "Asegúrate de que el bloque existe antes de conectarlo.",
              example: "Crea el bloque primero, luego conéctalo."
            });
          }
        }
      }
    }

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    };
  }

  private detectCycles(statements: Statement[]): boolean {
    const adj = new Map<string, string[]>();
    const inDegree = new Map<string, number>();
    const allIds = new Set<string>();

    for (const stmt of statements) {
      allIds.add(stmt.id);
      inDegree.set(stmt.id, 0);
    }

    for (const stmt of statements) {
      if (stmt.type === "TransformStatement") {
        for (const input of stmt.inputs) {
          if (!allIds.has(input)) continue;
          if (!adj.has(input)) {
            adj.set(input, []);
          }
          adj.get(input)!.push(stmt.id);
          inDegree.set(stmt.id, (inDegree.get(stmt.id) || 0) + 1);
        }
      } else if (stmt.type === "OutputStatement") {
        if (!allIds.has(stmt.input)) continue;
        if (!adj.has(stmt.input)) {
          adj.set(stmt.input, []);
        }
        adj.get(stmt.input)!.push(stmt.id);
        inDegree.set(stmt.id, (inDegree.get(stmt.id) || 0) + 1);
      }
    }

    const queue: string[] = [];
    for (const [id, degree] of inDegree.entries()) {
      if (degree === 0) {
        queue.push(id);
      }
    }

    let visited = 0;
    while (queue.length > 0) {
      const node = queue.shift()!;
      visited++;

      if (adj.has(node)) {
        for (const neighbor of adj.get(node)!) {
          inDegree.set(neighbor, inDegree.get(neighbor)! - 1);
          if (inDegree.get(neighbor) === 0) {
            queue.push(neighbor);
          }
        }
      }
    }

    return visited !== allIds.size;
  }

  private generateExample(operation: string, arity: number): string {
    if (operation === "ADD") {
      return "[5] + [3] ✅";
    }
    if (operation === "FILTER") {
      return "[formas] 🎨 [rojo] ✅";
    }
    return `Conecta ${arity} bloques al bloque "${operation}" ✅`;
  }

  private getInputType(identifier: string, typeTable: Map<string, string>): string {
    const inferredType = typeTable.get(identifier);
    if (!inferredType) {
      return "unknown";
    }
    return inferredType;
  }

  private isTypeCompatible(
    actualType: string,
    expectedType: DataType | TypeConstraint,
    operation: string,
    inputIndex: number
  ): boolean {
    if (typeof expectedType === "string") {
      return actualType === expectedType;
    }

    if (expectedType.kind === "set" || expectedType.kind === "stream") {
      const actualElementType = this.extractElementType(actualType);
      const expectedElementType = expectedType.elementType;

      if (actualElementType !== expectedElementType) {
        return false;
      }

      if (expectedType.kind === "set") {
        return this.validatePropertyConstraints(expectedType.elementType, operation);
      }

      return true;
    }

    return actualType === expectedType.kind;
  }

  private extractElementType(compositeType: string): string {
    const match = compositeType.match(/^(set|stream)<(.+)>$/);
    if (!match) {
      return "unknown";
    }
    return match[1];
  }

  private validatePropertyConstraints(elementType: DataType | string, operation: string): boolean {
    const propertyConstraints: Record<string, string[]> = {
      "FILTER_BY_COLOR": ["color"],
      "FILTER_BY_SIZE": ["size"],
      "FILTER_BY_TYPE": ["type"],
      "FILTER_BY_TASTE": ["taste"],
      "FILTER_BY_AGE_GROUP": ["ageGroup"],
      "FILTER_BY_GENDER": ["gender"],
      "COMPARE_BY_COLOR": ["color"],
      "COMPARE_BY_SIZE": ["size"],
      "COMPARE_BY_TYPE": ["type"],
      "COMPARE_BY_TASTE": ["taste"],
      "COMPARE_BY_AGE_GROUP": ["ageGroup"],
      "COMPARE_BY_GENDER": ["gender"]
    };

    const requiredProperties = propertyConstraints[operation];
    if (!requiredProperties) {
      return true;
    }

    for (const prop of requiredProperties) {
      if (!this.typeHasProperty(typeof elementType === "string" ? elementType : elementType.kind, prop)) {
        return false;
      }
    }

    return true;
  }

  private typeHasProperty(type: string, property: string): boolean {
    const curriculumTypes: Record<string, string[]> = {
      "shape": ["size", "color"],
      "car": ["color"],
      "food": ["taste", "color"],
      "animal": ["type", "color"],
      "person": ["ageGroup", "gender"]
    };

    const properties = curriculumTypes[type];
    return properties ? properties.includes(property) : false;
  }

  private generateTypeError(
    operation: string,
    expectedType: DataType | TypeConstraint,
    actualType: string,
    inputId: string,
    inputIndex: number
  ): ValidationError {
    const expectedTypeName = typeof expectedType === "string" ? expectedType : expectedType.kind;
    const error: ValidationError = {
      code: "TYPE_ERROR",
      message: "",
      nodeId: inputId,
      childMessage: "",
      suggestion: "",
      example: ""
    };

    if (operation === "ADD") {
      const isNumeric = actualType === "natural" || actualType === "integer" || actualType === "decimal";
      const hasText = actualType === "text";

      if (!isNumeric && hasText) {
        error.childMessage = `⚠️ ¡Ups! El bloque "${operation}" necesita números, no palabras. "${actualType}" no es un número.`;
        error.suggestion = `Conecta bloques de números (natural, integer, decimal) al bloque "${operation}".`;
        error.example = `[número 5] + [número 3] ✅`;
      } else if (!isNumeric) {
        error.childMessage = `⚠️ ¡Ups! El bloque "${operation}" necesita ${expectedTypeName}, recibiste "${actualType}".`;
        error.suggestion = `Conecta un bloque de tipo ${expectedTypeName}.`;
        error.example = `[${expectedTypeName} 5] + [${expectedTypeName} 3] ✅`;
      } else {
        error.childMessage = `⚠️ ¡Ups! El bloque "${operation}" recibió ${actualType} pero esperaba ${expectedTypeName}. Verifica tus conexiones.`;
        error.suggestion = `Revisa que "${inputId}" está conectado a un bloque de tipo ${expectedTypeName}.`;
        error.example = this.generateExample(operation, 2);
      }
    } else if (operation.startsWith("FILTER_BY_") || operation.startsWith("COMPARE_BY_")) {
      const property = operation.split("_BY_")[1].toLowerCase();
      const expectedTypeWithProperty = typeof expectedType === "object" ? Object.keys(expectedType).includes("elementType") ? `${expectedType.kind}<${(expectedType as any).elementType}>` : `${expectedType.kind}<${(expectedType as TypeConstraint).property}>` : expectedType;
      const elementType = this.extractElementType(actualType);

      error.message = `Operation ${operation} requires elements with '${property}' property, got ${actualType}`;
      error.childMessage = `⚠️ ¡Ups! El bloque "${operation}" necesita elementos con propiedad "${property}". "${elementType}" no tiene "${property}".`;
      error.suggestion = `Usa un tipo que tenga propiedad "${property}" (ej. ${expectedTypeWithProperty}).`;
      error.example = `${expectedTypeWithProperty} [elemento con ${property}] → filtrados ✅`;
    } else {
      error.childMessage = `⚠️ ¡Ups! El bloque "${operation}" espera ${expectedTypeName}, recibiste "${actualType}".`;
      error.suggestion = `Revisa el tipo de "${inputId}". Debe ser ${expectedTypeName}.`;
      error.example = `${expectedTypeName} → [${actualType}] ❌`;
    }

    return error;
  }
}

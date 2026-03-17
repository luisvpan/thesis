import type { DataflowNode, DataflowEdge, DataType, DataValue, TypeExpression } from "@dataflow/shared/types";
import type { ValidationError, ValidationResult } from "@dataflow/shared/types";
import type { Statement } from "../ast/ast-types.js";
import { OPERATION_REGISTRY, TypeConstraint, resolveOperationSignature } from "@dataflow/shared/operations";
import type { Curriculum, DataflowProgram } from "@dataflow/shared/types";

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
          childMessage: `⚠️ ¡Ups! Ya tienes un bloque llamado "${stmt.id}". Cada bloque necesita un nombre único.`,
          suggestion: `Cambia el nombre de uno de los bloques "${stmt.id}"`,
          example: `[${stmt.id}_1] y [${stmt.id}_2] ✅`
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

        const setHomogeneityErrors = this.validateSetHomogeneity(stmt);
        errors.push(...setHomogeneityErrors);

        const literalTypeErrors = this.validateLiteralType(stmt);
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
              childMessage: `⚠️ ¡Ups! No encuentro el bloque "${ref}".`,
              suggestion: "Asegúrate de que el bloque existe antes de conectarlo.",
              example: "Crea el bloque primero, luego conéctalo."
            });
          }
        }
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
        const signatures = OPERATION_REGISTRY[stmt.operation];
        if (!signatures) {
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

        const inputTypes = stmt.inputs.map(input => this.getInputType(input, typeTable));

        const arities = new Set(signatures.contracts.map(c => c.arity));
        if (!arities.has(stmt.inputs.length)) {
          const expectedArity = Array.from(arities).sort((a, b) => a - b)[0];
          errors.push({
            code: "WRONG_ARITY",
            message: `Operation ${stmt.operation} requires ${expectedArity} ${expectedArity === 1 ? 'input' : 'inputs'}, got ${stmt.inputs.length}`,
            nodeId: stmt.id,
            childMessage: `⚠️ ¡Ups! El bloque "${stmt.operation}" necesita ${expectedArity} ${expectedArity === 1 ? 'entrada' : 'entradas'}. Solo conectaste ${stmt.inputs.length}.`,
            suggestion: `Conecta ${expectedArity - stmt.inputs.length} bloque${expectedArity - stmt.inputs.length === 1 ? '' : 's'} más al bloque "${stmt.operation}".`,
            example: this.generateExample(stmt.operation, expectedArity)
          });
          continue;
        }

        const inputDataTypes = inputTypes.map(t => this.stringToTypeExpression(t)).filter((t): t is TypeExpression => t !== null) as TypeExpression[];

        const contract = resolveOperationSignature(stmt.operation, inputDataTypes);

        if (!contract) {
          const numericOps = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"];
          const hasNumericInput = inputTypes.some(t => ["natural", "integer", "decimal", "fraction"].includes(t));
          const hasTextInput = inputTypes.includes("text");
 
          let childMessage = `⚠️ ¡Ups! El bloque "${stmt.operation}" no funciona con estos tipos.`;
          let suggestion = `Verifica que los tipos de entrada sean compatibles con el bloque "${stmt.operation}".`;
          let example = `Usa tipos compatibles: [${signatures.contracts.map(c => c.inputTypes.join(", ")).join(" o ")}]`;
 
          if (stmt.operation === "SORT") {
            const isNumeric = inputTypes.every(t => ["natural", "integer", "decimal", "fraction"].includes(t));
            childMessage = `⚠️ ¡Ups! SORT solo funciona con números (natural, integer, decimal, fraction).\nTu conjunto contiene "${inputTypes.join(", ")}".`;
            suggestion = `Conecta bloques de números al bloque SORT. Para ordenar palabras, usa ALPHABETICAL_SORT.`;
            example = `[números 1, 5, 3] → SORT → [1, 3, 5] ✅`;
          } else if (stmt.operation === "ALPHABETICAL_SORT") {
            const isText = inputTypes.every(t => t === "text");
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

          const inputDataType = this.stringToDataType(inputType);
          if (!inputDataType) {
            continue;
          }

          if (!this.isTypeCompatible(inputType, expectedType, stmt.operation, i)) {
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
      const hasOutputNode = statements.some(stmt => stmt.type === "OutputStatement");
      if (!hasOutputNode) {
        errors.push({
          code: "MISSING_OUTPUT_NODE",
          message: "Program must have at least one output node",
          childMessage: "⚠️ ¡Ups! Tu programa necesita al menos un bloque de salida (output).",
          suggestion: "Añade un bloque de salida para ver el resultado de tu programa.",
          example: "[resultado] → [output] ✅"
        });
      }
    }

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    };
  }

  validateSetHomogeneity(stmt: Statement): ValidationError[] {
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
      const elementType = this.inferType(element, declaredElementType);

      if (elementType && elementType !== declaredElementType) {
        return [{
          code: "SET_HETEROGENEITY_ERROR",
          message: "All elements in set must have same type",
          nodeId: stmt.id,
          childMessage: "⚠️ ¡Ups! Un conjunto solo puede tener elementos del mismo tipo.",
          suggestion: `Asegúrate de que todos los elementos sean de tipo ${declaredElementType}.`,
          example: `set<${declaredElementType}> = {elemento 1, elemento 2} ✅`
        }];
      }
    }

    return [];
  }

  validateLiteralType(stmt: Statement): ValidationError[] {
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
            const valueType = this.inferType(elementValue, elementType);

            if (valueType && valueType !== elementType) {
              return [{
                code: "LITERAL_TYPE_MISMATCH",
                message: `Set element at index ${i} has type ${valueType}, but set expects ${elementType}`,
                nodeId: stmt.id,
                childMessage: `⚠️ ¡Ups! El elemento en posición ${i} no coincide con el tipo ${elementType}.`,
                suggestion: `Cambia el elemento o usa el bloque de tipo correcto.`,
                example: `Para "set<${elementType}>", usa elementos de tipo ${elementType}`
              }];
            }
          }
        }
        return [];
      }

    const valueType = this.inferType(stmt.value);
    if (valueType && valueType !== declaredType) {
      return [{
        code: "LITERAL_TYPE_MISMATCH",
        message: `Literal value ${JSON.stringify(stmt.value)} has type ${valueType}, but declared type is ${declaredType}`,
        nodeId: stmt.id,
        childMessage: "⚠️ ¡Ups! El valor no coincide con el tipo.",
        suggestion: `Cambia el valor o usa el bloque de tipo correcto.`,
        example: `Para "${declaredType}", usa un valor de tipo ${declaredType}`
      }];
    }

    return [];
  }

  private inferType(value: unknown, expectedType?: string): string | null {
    if (value === null || value === undefined) {
      return null;
    }

    if (typeof value === "number") {
      const result = Number.isInteger(value)
        ? (value >= 0
            ? (expectedType === "integer" ? "integer" : "natural")
            : "integer")
        : "decimal";
      return result;
    }

    if (typeof value === "string") {
      return "text";
    }

    if (typeof value === "boolean") {
      return "boolean";
    }

    if (typeof value === "object") {
      if (!value) {
        return null;
      }

      if ("kind" in value) {
        const kind = (value as any).kind;
        if (["natural", "integer", "decimal", "fraction", "text", "boolean"].includes(kind)) {
          return kind;
        }
        if (["shape", "car", "food", "animal", "person"].includes(kind)) {
          return kind;
        }
      }
    }

    return null;
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

  private stringToTypeExpression(typeStr: string): TypeExpression | null {
    if (!typeStr) return null;

    const setMatch = typeStr.match(/^set<(.+)>$/);
    if (setMatch) {
      const elementType = setMatch[1] as DataType;
      return { kind: "set", elementType };
    }

    const streamMatch = typeStr.match(/^stream<(.+)>$/);
    if (streamMatch) {
      const elementType = streamMatch[1] as DataType;
      return { kind: "stream", elementType };
    }

    return typeStr as DataType;
  }

  private stringToDataType(typeStr: string): DataValue | null {
    if (!typeStr) return null;

    if (typeStr === "natural") {
      return { kind: "natural", value: 0 };
    }
    if (typeStr === "integer") {
      return { kind: "integer", value: 0 };
    }
    if (typeStr === "decimal") {
      return { kind: "decimal", value: 0 };
    }
    if (typeStr === "fraction") {
      return { kind: "fraction", numerator: 0, denominator: 1 };
    }
    if (typeStr === "text") {
      return { kind: "text", value: "" };
    }
    if (typeStr === "boolean") {
      return { kind: "boolean", value: false };
    }
    if (typeStr === "shape") {
      return { kind: "shape", type: "circle", size: "small", color: "red" };
    }
    if (typeStr === "car") {
      return { kind: "car", color: "red" };
    }
    if (typeStr === "food") {
      return { kind: "food", taste: "sweet", color: "red" };
    }
    if (typeStr === "animal") {
      return { kind: "animal", type: "dog", color: "red" };
    }
    if (typeStr === "person") {
      return { kind: "person", ageGroup: "child", gender: "male" };
    }

    const setMatch = typeStr.match(/^set<(.+)>$/);
    if (setMatch) {
      const elementType = setMatch[1] as DataType;
      return { kind: "set", elementType, elements: [] };
    }

    const streamMatch = typeStr.match(/^stream<(.+)>$/);
    if (streamMatch) {
      const elementType = streamMatch[1] as DataType;
      return { kind: "stream", elementType, generator: (function*() {})() as Generator<DataValue> };
    }

    return null;
  }

  private getInputType(identifier: string, typeTable: Map<string, string>): string {
    const inferredType = typeTable.get(identifier);
    if (!inferredType) {
      return "unknown";
    }
    return inferredType;
  }

  private isTypeCompatible(
    actualType: DataType | string,
    expectedType: TypeExpression | TypeConstraint,
    operation: string,
    inputIndex: number
  ): boolean {
    const actualKind = typeof actualType === "string" ? actualType : actualType;
    const actualTypeStr = typeof actualType === "string" ? actualType : actualType;
    const actualElementStr = this.extractElementType(actualTypeStr);

    if (typeof expectedType === "string") {
      return actualType === expectedType;
    }

    if (expectedType.kind === "set" || expectedType.kind === "stream") {
      const expectedElementStr = typeof expectedType.elementType === "string" ? expectedType.elementType : expectedType.elementType;

      if (actualElementStr !== expectedElementStr) {
        return false;
      }

      if (expectedType.kind === "set") {
        return this.validatePropertyConstraints(expectedElementStr, operation);
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
    return match[2];
  }

  private validatePropertyConstraints(elementType: string, operation: string): boolean {
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
      if (!this.typeHasProperty(elementType, prop)) {
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
    expectedType: TypeExpression | TypeConstraint,
    actualType: string,
    inputId: string,
    inputIndex: number,
    transformNodeId: string
  ): ValidationError {
    const expectedTypeName = typeof expectedType === "string" ? expectedType : expectedType.kind;
    const error: ValidationError = {
      code: "TYPE_ERROR",
      message: `Operation ${operation} expects ${expectedTypeName}, received ${actualType}`,
      nodeId: transformNodeId,
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
    } else if (operation === "SORT") {
      const isNumeric = actualType === "natural" || actualType === "integer" || actualType === "decimal" || actualType === "fraction";
      
      if (!isNumeric) {
        error.childMessage = `⚠️ ¡Ups! SORT solo funciona con números (natural, integer, decimal, fraction).\nTu conjunto contiene "${actualType}".`;
        error.suggestion = `Conecta bloques de números al bloque SORT. Para ordenar palabras, usa ALPHABETICAL_SORT.`;
        error.example = `[números 1, 5, 3] → SORT → [1, 3, 5] ✅`;
      } else {
        error.childMessage = `⚠️ ¡Ups! El bloque SORT recibió ${actualType} pero esperaba un número.`;
        error.suggestion = `Revisa que "${inputId}" contiene solo números.`;
        error.example = `[números] → SORT ✅`;
      }
    } else if (operation === "ALPHABETICAL_SORT") {
      const isText = actualType === "text";
      
      if (!isText) {
        error.childMessage = `⚠️ ¡Ups! ALPHABETICAL_SORT solo funciona con texto (text).\nTu conjunto contiene "${actualType}".`;
        error.suggestion = `Conecta bloques de texto al bloque ALPHABETICAL_SORT. Para ordenar números, usa SORT.`;
        error.example = `[texto "z", "a", "m"] → ALPHABETICAL_SORT → ["a", "m", "z"] ✅`;
      } else {
        error.childMessage = `⚠️ ¡Ups! El bloque ALPHABETICAL_SORT recibió ${actualType} pero esperaba texto.`;
        error.suggestion = `Revisa que "${inputId}" contiene solo texto.`;
        error.example = `[texto] → ALPHABETICAL_SORT ✅`;
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

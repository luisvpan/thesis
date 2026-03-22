import type { DataType, DataValue, TypeExpression } from "@dataflow/shared/types";
import type { TypeConstraint } from "@dataflow/shared/operations";

export function inferType(value: unknown, expectedType?: string): string | null {
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

export function stringToTypeExpression(typeStr: string): TypeExpression | null {
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

export function stringToDataType(typeStr: string): DataValue | null {
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

export function getInputType(identifier: string, typeTable: Map<string, string>): string {
  const inferredType = typeTable.get(identifier);
  if (!inferredType) {
    return "unknown";
  }
  return inferredType;
}

export function isTypeCompatible(
  actualType: DataType | string,
  expectedType: TypeExpression | TypeConstraint,
  operation: string,
  inputIndex: number,
  typeHasProperty: (type: string, property: string) => boolean
): boolean {
  const actualTypeStr = typeof actualType === "string" ? actualType : actualType;
  const actualElementStr = extractElementType(actualTypeStr);

  if (typeof expectedType === "string") {
    return actualType === expectedType;
  }

  if (expectedType.kind === "set" || expectedType.kind === "stream") {
    const expectedElementStr = typeof expectedType.elementType === "string" ? expectedType.elementType : expectedType.elementType;

    if (actualElementStr !== expectedElementStr) {
      return false;
    }

    if (expectedType.kind === "set") {
      return validatePropertyConstraints(expectedElementStr, operation, typeHasProperty);
    }

    return true;
  }

  return actualType === expectedType.kind;
}

export function extractElementType(compositeType: string): string {
  const match = compositeType.match(/^(set|stream)<(.+)>$/);
  if (!match) {
    return "unknown";
  }
  return match[2];
}

export function validatePropertyConstraints(
  elementType: string,
  operation: string,
  typeHasProperty: (type: string, property: string) => boolean
): boolean {
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
    if (!typeHasProperty(elementType, prop)) {
      return false;
    }
  }

  return true;
}

export function typeHasProperty(type: string, property: string): boolean {
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

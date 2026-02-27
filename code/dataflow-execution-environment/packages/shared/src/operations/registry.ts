import type { DataType } from "../types/composite.js";

export type Operation =
  | "ADD"
  | "SUBTRACT"
  | "MULTIPLY"
  | "DIVIDE"
  | "COMPARE"
  | "FILTER"
  | "UNION"
  | "INTERSECTION"
  | "DIFFERENCE"
  | "COMPLEMENT"
  | "NEXT"
  | "FIRST"
  | "FBY"
  | "ACCUMULATE"
  | "SORT";

export type TypeConstraint = { kind: "hasProperty"; property: string };

export type OperationSignature = {
  arity: number;
  inputTypes: (DataType | TypeConstraint)[];
  outputType: DataType | ((inputs: DataType[]) => DataType);
  category: string;
};

export const OPERATION_REGISTRY: Record<string, OperationSignature> = {
  ADD: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "natural",
    category: "numeric"
  },

  SUBTRACT: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "integer",
    category: "numeric"
  },

  MULTIPLY: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "natural",
    category: "numeric"
  },

  DIVIDE: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "decimal",
    category: "numeric"
  },

  COMPARE: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "integer",
    category: "comparison"
  },

  FILTER: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "natural" } as DataType, "natural"],
    outputType: { kind: "set", elementType: "natural" } as DataType,
    category: "filtering"
  },

  UNION: {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "natural" } as DataType,
      { kind: "set", elementType: "natural" } as DataType
    ],
    outputType: { kind: "set", elementType: "natural" } as DataType,
    category: "sets"
  },

  INTERSECTION: {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "natural" } as DataType,
      { kind: "set", elementType: "natural" } as DataType
    ],
    outputType: { kind: "set", elementType: "natural" } as DataType,
    category: "sets"
  },

  DIFFERENCE: {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "natural" } as DataType,
      { kind: "set", elementType: "natural" } as DataType
    ],
    outputType: { kind: "set", elementType: "natural" } as DataType,
    category: "sets"
  },

  COMPLEMENT: {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "natural" } as DataType,
      { kind: "set", elementType: "natural" } as DataType
    ],
    outputType: { kind: "set", elementType: "natural" } as DataType,
    category: "sets"
  },

  NEXT: {
    arity: 1,
    inputTypes: [{ kind: "stream", elementType: "natural" } as DataType],
    outputType: "natural",
    category: "temporal"
  },

  FIRST: {
    arity: 1,
    inputTypes: [{ kind: "stream", elementType: "natural" } as DataType],
    outputType: "natural",
    category: "temporal"
  },

  FBY: {
    arity: 2,
    inputTypes: ["natural", { kind: "stream", elementType: "natural" } as DataType],
    outputType: { kind: "stream", elementType: "natural" } as DataType,
    category: "temporal"
  },

  ACCUMULATE: {
    arity: 3,
    inputTypes: [
      { kind: "stream", elementType: "natural" } as DataType,
      "natural",
      "natural"
    ],
    outputType: { kind: "stream", elementType: "natural" } as DataType,
    category: "temporal"
  },

  SORT: {
    arity: 1,
    inputTypes: [{ kind: "set", elementType: "natural" } as DataType],
    outputType: { kind: "set", elementType: "natural" } as DataType,
    category: "ordering"
  }
};

export function getOperationSignature(operation: string): OperationSignature | undefined {
  return OPERATION_REGISTRY[operation as Operation];
}

export function isOperation(name: string): boolean {
  return name in OPERATION_REGISTRY;
}

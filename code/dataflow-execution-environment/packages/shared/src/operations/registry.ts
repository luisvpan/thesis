import type { DataType } from "../types/composite.js";

export type Operation =
  | "ADD"
  | "SUBTRACT"
  | "MULTIPLY"
  | "DIVIDE"
  | "COMPARE"
  | "COMPARE_BY_SIZE"
  | "COMPARE_BY_COLOR"
  | "COMPARE_BY_TYPE"
  | "COMPARE_BY_TASTE"
  | "COMPARE_BY_AGE_GROUP"
  | "COMPARE_BY_GENDER"
  | "FILTER"
  | "FILTER_BY_SIZE"
  | "FILTER_BY_COLOR"
  | "FILTER_BY_TYPE"
  | "FILTER_BY_TASTE"
  | "FILTER_BY_AGE_GROUP"
  | "FILTER_BY_GENDER"
  | "UNION"
  | "INTERSECTION"
  | "DIFFERENCE"
  | "COMPLEMENT"
  | "NEXT"
  | "FIRST"
  | "FBY"
  | "ACCUMULATE"
  | "SORT"
  | "ALPHABETICAL_SORT"
  | "AND"
  | "OR"
  | "NOT";

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
    outputType: "boolean",
    category: "comparison"
  },

  COMPARE_BY_SIZE: {
    arity: 2,
    inputTypes: ["shape", "shape"],
    outputType: "boolean",
    category: "comparison"
  },

  COMPARE_BY_COLOR: {
    arity: 2,
    inputTypes: ["shape", "shape"],
    outputType: "boolean",
    category: "comparison"
  },

  COMPARE_BY_TYPE: {
    arity: 2,
    inputTypes: ["shape", "shape"],
    outputType: "boolean",
    category: "comparison"
  },

  COMPARE_BY_TASTE: {
    arity: 2,
    inputTypes: ["food", "food"],
    outputType: "boolean",
    category: "comparison"
  },

  COMPARE_BY_AGE_GROUP: {
    arity: 2,
    inputTypes: ["person", "person"],
    outputType: "boolean",
    category: "comparison"
  },

  COMPARE_BY_GENDER: {
    arity: 2,
    inputTypes: ["person", "person"],
    outputType: "boolean",
    category: "comparison"
  },

  FILTER: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "natural" } as DataType, "natural"],
    outputType: { kind: "set", elementType: "natural" } as DataType,
    category: "filtering"
  },

  FILTER_BY_SIZE: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "shape" } as DataType, "text"],
    outputType: { kind: "set", elementType: "shape" } as DataType,
    category: "filtering"
  },

  FILTER_BY_COLOR: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "shape" } as DataType, "text"],
    outputType: { kind: "set", elementType: "shape" } as DataType,
    category: "filtering"
  },

  FILTER_BY_TYPE: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "shape" } as DataType, "text"],
    outputType: { kind: "set", elementType: "shape" } as DataType,
    category: "filtering"
  },

  FILTER_BY_TASTE: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "food" } as DataType, "text"],
    outputType: { kind: "set", elementType: "food" } as DataType,
    category: "filtering"
  },

  FILTER_BY_AGE_GROUP: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "person" } as DataType, "text"],
    outputType: { kind: "set", elementType: "person" } as DataType,
    category: "filtering"
  },

  FILTER_BY_GENDER: {
    arity: 2,
    inputTypes: [{ kind: "set", elementType: "person" } as DataType, "text"],
    outputType: { kind: "set", elementType: "person" } as DataType,
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
  },

  ALPHABETICAL_SORT: {
    arity: 1,
    inputTypes: [{ kind: "set", elementType: "text" } as DataType],
    outputType: { kind: "set", elementType: "text" } as DataType,
    category: "ordering"
  },

  AND: {
    arity: 2,
    inputTypes: ["boolean", "boolean"],
    outputType: "boolean",
    category: "boolean"
  },

  OR: {
    arity: 2,
    inputTypes: ["boolean", "boolean"],
    outputType: "boolean",
    category: "boolean"
  },

  NOT: {
    arity: 1,
    inputTypes: ["boolean"],
    outputType: "boolean",
    category: "boolean"
  }
};

export function getOperationSignature(operation: string): OperationSignature | undefined {
  return OPERATION_REGISTRY[operation as Operation];
}

export function isOperation(name: string): boolean {
  return name in OPERATION_REGISTRY;
}

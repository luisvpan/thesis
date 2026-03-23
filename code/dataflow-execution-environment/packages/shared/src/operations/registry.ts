import type { DataType, TypeExpression } from "../types/composite.js";

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
  inputTypes: (TypeExpression | TypeConstraint)[];
  outputType: TypeExpression | ((inputs: TypeExpression[]) => TypeExpression);
  category: string;
};

export type OperationContract = {
  arity: number;
  inputTypes: (TypeExpression | TypeConstraint)[];
  outputType: TypeExpression;
  category: string;
};

export type OperationSignatures = {
  contracts: OperationContract[];
  category: string;
};

export const OPERATION_REGISTRY: Record<string, OperationSignatures> = {
  ADD: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "natural", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" }
    ],
    category: "numeric"
  },

  SUBTRACT: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" }
    ],
    category: "numeric"
  },

  MULTIPLY: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "natural", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" }
    ],
    category: "numeric"
  },

  DIVIDE: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" }
    ],
    category: "numeric"
  },

  COMPARE: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["text", "text"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["boolean", "boolean"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  COMPARE_BY_SIZE: {
    contracts: [
      { arity: 2, inputTypes: ["shape", "shape"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  COMPARE_BY_COLOR: {
    contracts: [
      { arity: 2, inputTypes: ["shape", "shape"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["car", "car"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["food", "food"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["animal", "animal"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  COMPARE_BY_TYPE: {
    contracts: [
      { arity: 2, inputTypes: ["shape", "shape"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["animal", "animal"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  COMPARE_BY_TASTE: {
    contracts: [
      { arity: 2, inputTypes: ["food", "food"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  COMPARE_BY_AGE_GROUP: {
    contracts: [
      { arity: 2, inputTypes: ["person", "person"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  COMPARE_BY_GENDER: {
    contracts: [
      { arity: 2, inputTypes: ["person", "person"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  FILTER: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" }, "natural"], outputType: { kind: "set", elementType: "natural" }, category: "filtering" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "integer" }, "integer"], outputType: { kind: "set", elementType: "integer" }, category: "filtering" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "decimal" }, "decimal"], outputType: { kind: "set", elementType: "decimal" }, category: "filtering" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "fraction" }, "fraction"], outputType: { kind: "set", elementType: "fraction" }, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_SIZE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" }, "text"], outputType: { kind: "set", elementType: "shape" }, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_COLOR: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" }, "text"], outputType: { kind: "set", elementType: "shape" }, category: "filtering" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "car" }, "text"], outputType: { kind: "set", elementType: "car" }, category: "filtering" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "food" }, "text"], outputType: { kind: "set", elementType: "food" }, category: "filtering" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "animal" }, "text"], outputType: { kind: "set", elementType: "animal" }, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_TYPE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" }, "text"], outputType: { kind: "set", elementType: "shape" }, category: "filtering" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "animal" }, "text"], outputType: { kind: "set", elementType: "animal" }, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_TASTE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "food" }, "text"], outputType: { kind: "set", elementType: "food" }, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_AGE_GROUP: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" }, "text"], outputType: { kind: "set", elementType: "person" }, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_GENDER: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" }, "text"], outputType: { kind: "set", elementType: "person" }, category: "filtering" }
    ],
    category: "filtering"
  },

  UNION: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" }, { kind: "set", elementType: "natural" }], outputType: { kind: "set", elementType: "natural" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" }, { kind: "set", elementType: "shape" }], outputType: { kind: "set", elementType: "shape" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "car" }, { kind: "set", elementType: "car" }], outputType: { kind: "set", elementType: "car" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "food" }, { kind: "set", elementType: "food" }], outputType: { kind: "set", elementType: "food" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "animal" }, { kind: "set", elementType: "animal" }], outputType: { kind: "set", elementType: "animal" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" }, { kind: "set", elementType: "person" }], outputType: { kind: "set", elementType: "person" }, category: "sets" }
    ],
    category: "sets"
  },

  INTERSECTION: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" }, { kind: "set", elementType: "natural" }], outputType: { kind: "set", elementType: "natural" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" }, { kind: "set", elementType: "shape" }], outputType: { kind: "set", elementType: "shape" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "car" }, { kind: "set", elementType: "car" }], outputType: { kind: "set", elementType: "car" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "food" }, { kind: "set", elementType: "food" }], outputType: { kind: "set", elementType: "food" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "animal" }, { kind: "set", elementType: "animal" }], outputType: { kind: "set", elementType: "animal" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" }, { kind: "set", elementType: "person" }], outputType: { kind: "set", elementType: "person" }, category: "sets" }
    ],
    category: "sets"
  },

  DIFFERENCE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" }, { kind: "set", elementType: "natural" }], outputType: { kind: "set", elementType: "natural" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" }, { kind: "set", elementType: "shape" }], outputType: { kind: "set", elementType: "shape" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "car" }, { kind: "set", elementType: "car" }], outputType: { kind: "set", elementType: "car" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "food" }, { kind: "set", elementType: "food" }], outputType: { kind: "set", elementType: "food" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "animal" }, { kind: "set", elementType: "animal" }], outputType: { kind: "set", elementType: "animal" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" }, { kind: "set", elementType: "person" }], outputType: { kind: "set", elementType: "person" }, category: "sets" }
    ],
    category: "sets"
  },

  COMPLEMENT: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" }, { kind: "set", elementType: "natural" }], outputType: { kind: "set", elementType: "natural" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" }, { kind: "set", elementType: "shape" }], outputType: { kind: "set", elementType: "shape" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "car" }, { kind: "set", elementType: "car" }], outputType: { kind: "set", elementType: "car" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "food" }, { kind: "set", elementType: "food" }], outputType: { kind: "set", elementType: "food" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "animal" }, { kind: "set", elementType: "animal" }], outputType: { kind: "set", elementType: "animal" }, category: "sets" },
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" }, { kind: "set", elementType: "person" }], outputType: { kind: "set", elementType: "person" }, category: "sets" }
    ],
    category: "sets"
  },

  NEXT: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "natural" }], outputType: "natural", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "integer" }], outputType: "integer", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "decimal" }], outputType: "decimal", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "fraction" }], outputType: "fraction", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "text" }], outputType: "text", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "boolean" }], outputType: "boolean", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "shape" }], outputType: "shape", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "car" }], outputType: "car", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "food" }], outputType: "food", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "animal" }], outputType: "animal", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "person" }], outputType: "person", category: "temporal" }
    ],
    category: "temporal"
  },

  FIRST: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "natural" }], outputType: "natural", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "integer" }], outputType: "integer", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "decimal" }], outputType: "decimal", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "fraction" }], outputType: "fraction", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "text" }], outputType: "text", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "boolean" }], outputType: "boolean", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "shape" }], outputType: "shape", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "car" }], outputType: "car", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "food" }], outputType: "food", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "animal" }], outputType: "animal", category: "temporal" },
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "person" }], outputType: "person", category: "temporal" }
    ],
    category: "temporal"
  },

  FBY: {
    contracts: [
      { arity: 2, inputTypes: ["natural", { kind: "stream", elementType: "natural" }], outputType: { kind: "stream", elementType: "natural" }, category: "temporal" },
      { arity: 2, inputTypes: ["integer", { kind: "stream", elementType: "integer" }], outputType: { kind: "stream", elementType: "integer" }, category: "temporal" },
      { arity: 2, inputTypes: ["decimal", { kind: "stream", elementType: "decimal" }], outputType: { kind: "stream", elementType: "decimal" }, category: "temporal" },
      { arity: 2, inputTypes: ["fraction", { kind: "stream", elementType: "fraction" }], outputType: { kind: "stream", elementType: "fraction" }, category: "temporal" },
      { arity: 2, inputTypes: ["text", { kind: "stream", elementType: "text" }], outputType: { kind: "stream", elementType: "text" }, category: "temporal" },
      { arity: 2, inputTypes: ["boolean", { kind: "stream", elementType: "boolean" }], outputType: { kind: "stream", elementType: "boolean" }, category: "temporal" },
      { arity: 2, inputTypes: ["shape", { kind: "stream", elementType: "shape" }], outputType: { kind: "stream", elementType: "shape" }, category: "temporal" },
      { arity: 2, inputTypes: ["car", { kind: "stream", elementType: "car" }], outputType: { kind: "stream", elementType: "car" }, category: "temporal" },
      { arity: 2, inputTypes: ["food", { kind: "stream", elementType: "food" }], outputType: { kind: "stream", elementType: "food" }, category: "temporal" },
      { arity: 2, inputTypes: ["animal", { kind: "stream", elementType: "animal" }], outputType: { kind: "stream", elementType: "animal" }, category: "temporal" },
      { arity: 2, inputTypes: ["person", { kind: "stream", elementType: "person" }], outputType: { kind: "stream", elementType: "person" }, category: "temporal" }
    ],
    category: "temporal"
  },

  ACCUMULATE: {
    contracts: [
      { arity: 3, inputTypes: [{ kind: "stream", elementType: "natural" }, "text", "natural"], outputType: { kind: "stream", elementType: "natural" }, category: "temporal" },
      { arity: 3, inputTypes: [{ kind: "stream", elementType: "integer" }, "text", "integer"], outputType: { kind: "stream", elementType: "integer" }, category: "temporal" },
      { arity: 3, inputTypes: [{ kind: "stream", elementType: "decimal" }, "text", "decimal"], outputType: { kind: "stream", elementType: "decimal" }, category: "temporal" },
      { arity: 3, inputTypes: [{ kind: "stream", elementType: "fraction" }, "text", "fraction"], outputType: { kind: "stream", elementType: "fraction" }, category: "temporal" }
    ],
    category: "temporal"
  },

  SORT: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "set", elementType: "natural" }], outputType: { kind: "set", elementType: "natural" }, category: "ordering" },
      { arity: 1, inputTypes: [{ kind: "set", elementType: "integer" }], outputType: { kind: "set", elementType: "integer" }, category: "ordering" },
      { arity: 1, inputTypes: [{ kind: "set", elementType: "decimal" }], outputType: { kind: "set", elementType: "decimal" }, category: "ordering" },
      { arity: 1, inputTypes: [{ kind: "set", elementType: "fraction" }], outputType: { kind: "set", elementType: "fraction" }, category: "ordering" }
    ],
    category: "ordering"
  },

  ALPHABETICAL_SORT: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "set", elementType: "text" }], outputType: { kind: "set", elementType: "text" }, category: "ordering" }
    ],
    category: "ordering"
  },

  AND: {
    contracts: [
      { arity: 2, inputTypes: ["boolean", "boolean"], outputType: "boolean", category: "boolean" }
    ],
    category: "boolean"
  },

  OR: {
    contracts: [
      { arity: 2, inputTypes: ["boolean", "boolean"], outputType: "boolean", category: "boolean" }
    ],
    category: "boolean"
  },

  NOT: {
    contracts: [
      { arity: 1, inputTypes: ["boolean"], outputType: "boolean", category: "boolean" }
    ],
    category: "boolean"
  }
};

export function getOperationSignatures(operation: string): OperationSignatures | undefined {
  return OPERATION_REGISTRY[operation as Operation];
}

export function resolveOperationSignature(operation: string, inputTypes: TypeExpression[]): OperationContract | undefined {
  const op = OPERATION_REGISTRY[operation as Operation];
  if (!op) return undefined;

  for (const contract of op.contracts) {
    if (contract.inputTypes.length !== inputTypes.length) continue;

    let matches = true;
    for (let i = 0; i < inputTypes.length; i++) {
      const expected = contract.inputTypes[i];
      const actual = inputTypes[i];

      if (!typesMatch(expected, actual)) {
        matches = false;
        break;
      }
    }

    if (matches) {
      return contract;
    }
  }

  return undefined;
}

function typesMatch(expected: TypeExpression | TypeConstraint, actual: TypeExpression): boolean {
  if (typeof expected === "string" && typeof actual === "string") {
    return expected === actual;
  }

  if (typeof expected === "string" && typeof actual === "object") {
    return false;
  }

  if (typeof expected === "object" && expected !== null) {
    if (expected.kind === "hasProperty") {
      return true;
    }
    if (expected.kind === "set" && typeof actual === "object" && actual !== null && actual.kind === "set") {
      return typesMatch(expected.elementType, actual.elementType);
    }
    if (expected.kind === "stream" && typeof actual === "object" && actual !== null && actual.kind === "stream") {
      return typesMatch(expected.elementType, actual.elementType);
    }
    return false;
  }

  return false;
}

export function isOperation(name: string): boolean {
  return name in OPERATION_REGISTRY;
}

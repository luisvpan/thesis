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

export type OperationContract = {
  arity: number;
  inputTypes: (DataType | TypeConstraint)[];
  outputType: DataType;
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
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "integer"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "natural"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "natural"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "integer"], outputType: "decimal", category: "numeric" }
    ],
    category: "numeric"
  },

  SUBTRACT: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "integer"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "natural"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "natural"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "integer"], outputType: "decimal", category: "numeric" }
    ],
    category: "numeric"
  },

  MULTIPLY: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "natural", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "integer", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "integer"], outputType: "natural", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "natural"], outputType: "natural", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "natural"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "integer"], outputType: "decimal", category: "numeric" }
    ],
    category: "numeric"
  },

  DIVIDE: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "integer"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "natural"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["natural", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "natural"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["integer", "decimal"], outputType: "decimal", category: "numeric" },
      { arity: 2, inputTypes: ["decimal", "integer"], outputType: "decimal", category: "numeric" }
    ],
    category: "numeric"
  },

  COMPARE: {
    contracts: [
      { arity: 2, inputTypes: ["natural", "natural"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["integer", "integer"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["decimal", "decimal"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["natural", "integer"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["integer", "natural"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["natural", "decimal"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["decimal", "natural"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["integer", "decimal"], outputType: "boolean", category: "comparison" },
      { arity: 2, inputTypes: ["decimal", "integer"], outputType: "boolean", category: "comparison" }
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
      { arity: 2, inputTypes: ["shape", "shape"], outputType: "boolean", category: "comparison" }
    ],
    category: "comparison"
  },

  COMPARE_BY_TYPE: {
    contracts: [
      { arity: 2, inputTypes: ["shape", "shape"], outputType: "boolean", category: "comparison" }
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
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" } as DataType, "natural"], outputType: { kind: "set", elementType: "natural" } as DataType, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_SIZE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" } as DataType, "text"], outputType: { kind: "set", elementType: "shape" } as DataType, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_COLOR: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" } as DataType, "text"], outputType: { kind: "set", elementType: "shape" } as DataType, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_TYPE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "shape" } as DataType, "text"], outputType: { kind: "set", elementType: "shape" } as DataType, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_TASTE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "food" } as DataType, "text"], outputType: { kind: "set", elementType: "food" } as DataType, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_AGE_GROUP: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" } as DataType, "text"], outputType: { kind: "set", elementType: "person" } as DataType, category: "filtering" }
    ],
    category: "filtering"
  },

  FILTER_BY_GENDER: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "person" } as DataType, "text"], outputType: { kind: "set", elementType: "person" } as DataType, category: "filtering" }
    ],
    category: "filtering"
  },

  UNION: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" } as DataType, { kind: "set", elementType: "natural" } as DataType], outputType: { kind: "set", elementType: "natural" } as DataType, category: "sets" }
    ],
    category: "sets"
  },

  INTERSECTION: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" } as DataType, { kind: "set", elementType: "natural" } as DataType], outputType: { kind: "set", elementType: "natural" } as DataType, category: "sets" }
    ],
    category: "sets"
  },

  DIFFERENCE: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" } as DataType, { kind: "set", elementType: "natural" } as DataType], outputType: { kind: "set", elementType: "natural" } as DataType, category: "sets" }
    ],
    category: "sets"
  },

  COMPLEMENT: {
    contracts: [
      { arity: 2, inputTypes: [{ kind: "set", elementType: "natural" } as DataType, { kind: "set", elementType: "natural" } as DataType], outputType: { kind: "set", elementType: "natural" } as DataType, category: "sets" }
    ],
    category: "sets"
  },

  NEXT: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "natural" } as DataType], outputType: "natural", category: "temporal" }
    ],
    category: "temporal"
  },

  FIRST: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "stream", elementType: "natural" } as DataType], outputType: "natural", category: "temporal" }
    ],
    category: "temporal"
  },

  FBY: {
    contracts: [
      { arity: 2, inputTypes: ["natural", { kind: "stream", elementType: "natural" } as DataType], outputType: { kind: "stream", elementType: "natural" } as DataType, category: "temporal" }
    ],
    category: "temporal"
  },

  ACCUMULATE: {
    contracts: [
      { arity: 3, inputTypes: [{ kind: "stream", elementType: "natural" } as DataType, "natural", "natural"], outputType: { kind: "stream", elementType: "natural" } as DataType, category: "temporal" }
    ],
    category: "temporal"
  },

  SORT: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "set", elementType: "natural" } as DataType], outputType: { kind: "set", elementType: "natural" } as DataType, category: "ordering" }
    ],
    category: "ordering"
  },

  ALPHABETICAL_SORT: {
    contracts: [
      { arity: 1, inputTypes: [{ kind: "set", elementType: "text" } as DataType], outputType: { kind: "set", elementType: "text" } as DataType, category: "ordering" }
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

export function resolveOperationSignature(operation: string, inputTypes: DataType[]): OperationContract | undefined {
  const op = OPERATION_REGISTRY[operation as Operation];
  if (!op) return undefined;

  for (const contract of op.contracts) {
    if (contract.inputTypes.length !== inputTypes.length) continue;
    
    let matches = true;
    for (let i = 0; i < inputTypes.length; i++) {
      const expected = contract.inputTypes[i];
      const actual = inputTypes[i];
      
      if (typeof expected === "string") {
        if (expected !== actual && typeof actual !== "object") {
          matches = false;
          break;
        }
        if (typeof actual === "object" && actual !== null) {
          const actualObj = actual as { kind: string };
          const expectedObj = expected as string;
          if (actualObj.kind !== expectedObj) {
            matches = false;
            break;
          }
        }
      }
    }
    
    if (matches) {
      return contract;
    }
  }
  
  return undefined;
}

export function isOperation(name: string): boolean {
  return name in OPERATION_REGISTRY;
}

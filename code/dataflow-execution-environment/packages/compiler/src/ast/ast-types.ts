import type { DataSourceNode, TransformationNode, OutputNode } from "@dataflow/shared/types";

export type SourceStatement = {
  type: "SourceStatement";
  id: string;
  dataType: string;
  value: unknown;
};

export type TransformStatement = {
  type: "TransformStatement";
  id: string;
  dataType: string;
  operation: string;
  inputs: string[];
};

export type OutputStatement = {
  type: "OutputStatement";
  id: string;
  dataType: string;
  input: string;
};

export type Statement = SourceStatement | TransformStatement | OutputStatement;

export type Program = {
  type: "Program";
  statements: Statement[];
};

export function statementToNode(stmt: Statement): DataSourceNode | TransformationNode | OutputNode {
  if (stmt.type === "SourceStatement") {
    return {
      id: stmt.id,
      type: "DataSource",
      dataType: stmt.dataType as any,
      value: stmt.value
    };
  } else if (stmt.type === "TransformStatement") {
    return {
      id: stmt.id,
      type: "Transformation",
      dataType: stmt.dataType as any,
      operation: stmt.operation,
      inputs: stmt.inputs
    };
  } else {
    return {
      id: stmt.id,
      type: "Output",
      dataType: stmt.dataType as any,
      input: stmt.input
    };
  }
}

import type { DataSourceNode, TransformationNode, OutputNode, DataflowNode, NodeMetadata } from "@dataflow/shared/types";

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

export function statementToNode(stmt: Statement, metadata?: NodeMetadata): DataflowNode {
  if (stmt.type === "SourceStatement") {
    return {
      id: stmt.id,
      type: "DataSource",
      dataType: stmt.dataType as any,
      value: stmt.value,
      metadata
    };
  } else if (stmt.type === "TransformStatement") {
    return {
      id: stmt.id,
      type: "Transformation",
      dataType: stmt.dataType as any,
      operation: stmt.operation,
      inputs: stmt.inputs,
      metadata
    };
  } else {
    return {
      id: stmt.id,
      type: "Output",
      dataType: stmt.dataType as any,
      input: stmt.input,
      metadata
    };
  }
}

export function nodeToStatement(node: DataflowNode): Statement {
  if (node.type === "DataSource") {
    return {
      type: "SourceStatement",
      id: node.id,
      dataType: node.dataType as any,
      value: (node as DataSourceNode).value
    };
  } else if (node.type === "Transformation") {
    return {
      type: "TransformStatement",
      id: node.id,
      dataType: node.dataType as any,
      operation: (node as TransformationNode).operation,
      inputs: (node as TransformationNode).inputs
    };
  } else {
    return {
      type: "OutputStatement",
      id: node.id,
      dataType: node.dataType as any,
      input: (node as OutputNode).input
    };
  }
}
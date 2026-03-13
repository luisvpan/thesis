import type { DataType } from "./composite.js";

export type NodeMetadata = {
  label?: string;
  blockId?: string;
  position?: [number, number];
  isLiteral?: boolean;
};

export type DataSourceNode = {
  id: string;
  type: "DataSource";
  dataType: DataType;
  value: unknown;
  metadata?: NodeMetadata;
};

export type TransformationNode = {
  id: string;
  type: "Transformation";
  dataType: DataType;
  operation: string;
  inputs: string[];
  metadata?: NodeMetadata;
};

export type OutputNode = {
  id: string;
  type: "Output";
  dataType: DataType;
  input: string;
  metadata?: NodeMetadata;
};

export type DataflowNode = DataSourceNode | TransformationNode | OutputNode;

export type DataflowEdge = {
  id: string;
  from: string;
  to: string;
  toPort?: number;
};

export type DataflowProgram = {
  metadata: {
    programId: string;
    activityId?: string;
    level?: number;
  };
  graph: {
    nodes: DataflowNode[];
    edges: DataflowEdge[];
  };
};

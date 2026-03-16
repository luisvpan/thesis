import type { Primitive } from "./primitives.js";
import type { Curriculum } from "./curriculum.js";

export type DataType =
  | Primitive["kind"]
  | Curriculum["kind"]
  | "set"
  | "stream";

export type TypeExpression = DataType | SetTypeExpression | StreamTypeExpression;

export type SetTypeExpression = {
  kind: "set";
  elementType: DataType;
};

export type StreamTypeExpression = {
  kind: "stream";
  elementType: DataType;
};

export type SetType = {
  kind: "set";
  elementType: DataType;
  elements: DataValue[];
};

export type StreamType = {
  kind: "stream";
  elementType: DataType;
  generator: Generator<DataValue>;
};

export type DataValue =
  | Primitive
  | Curriculum
  | SetType
  | StreamType;

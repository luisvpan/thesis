import type { Primitive } from "./primitives.js";
import type { Curriculum } from "./curriculum.js";

export type DataType =
  | Primitive["kind"]
  | Curriculum["kind"]
  | { kind: "set"; elementType: string | DataType }
  | { kind: "stream"; elementType: string | DataType }
  | "fraction";

export type SetType<T = DataType> = {
  kind: "set";
  elementType: T;
  elements: unknown[];
};

export type StreamType<T = DataType> = {
  kind: "stream";
  elementType: T;
  generator: Generator<T>;
};

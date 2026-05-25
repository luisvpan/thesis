export {
  Interpreter,
  type EvaluationStats,
  type ExecuteResult,
  type ParseError,
} from "./interpreter";

export { RuntimeError, type ErrorCode } from "./runtime/errors";

export { serialize, deserialize, type SerializeResult } from "./serializer";

export { formatValue } from "./formatter";

export type {
  Program,
  Statement,
  SourceStatement,
  TransformStatement,
  SinkStatement,
  Expression,
  IdentifierExpression,
  Literal,
  ArrayLiteral,
  OtherLiteral,
  ObjectLiteral,
  ObjectProperty,
  Operation,
} from "./program";

export {
  createAbstractNumber,
  createPictoricObject,
  createConcreteObject,
} from "./utils";

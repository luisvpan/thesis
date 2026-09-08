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
  // v4.0.0 discriminated types
  DataLiteral,
  CriteriaLiteral,
  // Legacy (deprecated)
  ObjectProperty,
  Operation,
} from "./program";

export {
  isDataLiteral,
  isCriteriaLiteral,
} from "./program";

export {
  // Runtime helpers
  createAbstractNumber,
  createPictoricObject,
  createConcreteObject,
  // AST helpers (v4.0.0)
  createAbstractDataLiteral,
  createPictoricDataLiteral,
  createConcreteDataLiteral,
  createCriteriaLiteral,
} from "./utils";

export {
  Interpreter,
  type EvaluationStats,
  type ExecuteResult,
  type ParseError,
} from "./interpreter";

export {
  RuntimeError,
  type ErrorCode,
  type ErrorPhase,
  type ErrorSite,
  type RuntimeErrorCode,
  type StaticErrorCode,
} from "./runtime/errors";

export { serialize, deserialize, type SerializeResult } from "./serializer";

export { formatValue } from "./formatter";

// Construcción de valores
export {
  createBag,
  createFilterCriterion,
  createOrderCriterion,
  ImmutableBag,
  type EntrySpec,
} from "./bag-builder";

// Estructura de un programa
export type {
  BagLiteral,
  CriterionLiteral,
  Expression,
  IdentifierExpression,
  Literal,
  Operation,
  Program,
  SinkStatement,
  SourceStatement,
  Statement,
  TransformStatement,
} from "./program";

export { isBagLiteral, isCriterionLiteral } from "./program";

// Modelo de valores
export type {
  Bag,
  BooleanValue,
  CPACategory,
  Criterion,
  CriterionSubtype,
  CriterionValue,
  Entry,
  OrderDirection,
  RuntimeValue,
  ValueCategory,
} from "./runtime/types";

export { isBag, isBoolean, isCriterion } from "./runtime/types";

export { OPERATIONS, isOperation } from "./operations/signatures";

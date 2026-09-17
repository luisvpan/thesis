export {
  Interpreter,
  type EvaluationStats,
  type ExecuteResult,
} from "./interpreter";

// Errores: una sola forma para las tres fases (§4). Los códigos se exportan
// también como valores, para poder recorrerlos o construir un mapa de mensajes.
export {
  DataflowError,
  ERROR_CODES,
  RUNTIME_ERROR_CODES,
  STATIC_ERROR_CODES,
  SYNTAX_ERROR_CODES,
  isDataflowError,
  type ErrorCode,
  type ErrorPhase,
  type RuntimeErrorCode,
  type StaticErrorCode,
  type SyntaxErrorCode,
} from "./runtime/errors";

export { serialize, deserialize, type SerializeResult } from "./serializer";

export { formatValue } from "./formatter";

// Construcción de programas
export { createProgram, ProgramBuilder } from "./program-builder";

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

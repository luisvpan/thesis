// Errores — LANGUAGE_SPEC.md §4
//
// Una sola forma para las tres fases. Todo error informa su naturaleza, el nodo
// donde ocurrió, el nodo que lo causó (si es otro) y, si surgió al evaluar, la
// salida en cuyo cálculo apareció.

/** El momento en que se detecta el error (§4). */
export type ErrorPhase = "syntax" | "static" | "runtime";

/**
 * Errores de sintaxis (§4.1). No se enumeran uno por uno: su especificación es
 * la gramática, así que lo que sitúa al error es el detalle y su posición.
 */
export const SYNTAX_ERROR_CODES = ["SYNTAX_ERROR"] as const;

/** Errores estáticos (§4.2): invalidan el programa completo. */
export const STATIC_ERROR_CODES = [
  "DUPLICATE_IDENTIFIER",
  "UNDEFINED_REFERENCE",
  "CIRCULAR_DEPENDENCY",
  "UNKNOWN_OPERATION",
  "ARITY_ERROR",
  "TYPE_ERROR",
  "INVALID_CRITERION",
  "INVALID_OBJECT",
] as const;

/** Errores de ejecución (§4.3): dependen del valor, aislados por salida. */
export const RUNTIME_ERROR_CODES = ["EXPECTED_NUMBER", "DIVISION_BY_ZERO"] as const;

export const ERROR_CODES = [
  ...SYNTAX_ERROR_CODES,
  ...STATIC_ERROR_CODES,
  ...RUNTIME_ERROR_CODES,
] as const;

export type SyntaxErrorCode = (typeof SYNTAX_ERROR_CODES)[number];
export type StaticErrorCode = (typeof STATIC_ERROR_CODES)[number];
export type RuntimeErrorCode = (typeof RUNTIME_ERROR_CODES)[number];
export type ErrorCode = (typeof ERROR_CODES)[number];

const STATIC_CODES = new Set<string>(STATIC_ERROR_CODES);
const SYNTAX_CODES = new Set<string>(SYNTAX_ERROR_CODES);

function phaseOf(code: ErrorCode): ErrorPhase {
  if (SYNTAX_CODES.has(code)) return "syntax";
  return STATIC_CODES.has(code) ? "static" : "runtime";
}

/** Dónde queda situado el error (§4). */
export interface ErrorSite {
  /** El nodo que se estaba procesando. */
  nodeId?: string;
  /** El nodo que causó la falla; coincide con `nodeId` si la falla es local. */
  causeNodeId?: string;
  /** La salida en cuyo cálculo bajo demanda apareció. */
  sinkId?: string;
  /**
   * Posición del argumento culpable, cuando una operación falla por el valor de
   * uno de ellos. El evaluador la traduce a `causeNodeId`, que es quien conoce
   * los nombres de los nodos.
   */
  argumentIndex?: number;
  /** Posición en el texto; solo la llevan los errores de sintaxis. */
  line?: number;
  column?: number;
}

export class DataflowError extends Error {
  readonly code: ErrorCode;
  readonly phase: ErrorPhase;
  /** El mensaje sin el prefijo de situación. */
  readonly detail: string;
  nodeId?: string;
  causeNodeId?: string;
  sinkId?: string;
  readonly argumentIndex?: number;
  readonly line?: number;
  readonly column?: number;

  constructor(code: ErrorCode, detail: string, site: ErrorSite = {}) {
    super(detail);
    this.name = "DataflowError";
    this.code = code;
    this.detail = detail;
    this.phase = phaseOf(code);
    this.nodeId = site.nodeId;
    this.causeNodeId = site.causeNodeId ?? site.nodeId;
    this.sinkId = site.sinkId;
    this.argumentIndex = site.argumentIndex;
    this.line = site.line;
    this.column = site.column;
    this.message = describe(this);
  }

  /**
   * Completa la situación del error con lo que sepa quien lo propaga, sin pisar
   * lo que ya estuviera fijado (el nodo donde ocurrió manda sobre el de quien
   * lo recibe).
   */
  situate(site: ErrorSite): this {
    this.nodeId ??= site.nodeId;
    this.causeNodeId ??= site.causeNodeId ?? this.nodeId;
    this.sinkId ??= site.sinkId;
    this.message = describe(this);
    return this;
  }
}

export function isDataflowError(value: unknown): value is DataflowError {
  return value instanceof DataflowError;
}

/** Un error de sintaxis, con su posición en el texto (§4.1). */
export function syntaxError(detail: string, position: { line?: number; column?: number } = {}): DataflowError {
  return new DataflowError("SYNTAX_ERROR", detail, position);
}

function describe(error: DataflowError): string {
  const parts: string[] = [];

  if (error.nodeId) parts.push(`en '${error.nodeId}'`);
  if (error.causeNodeId && error.causeNodeId !== error.nodeId) {
    parts.push(`por '${error.causeNodeId}'`);
  }
  if (error.sinkId) parts.push(`salida '${error.sinkId}'`);
  if (error.line !== undefined) {
    parts.push(error.column === undefined ? `línea ${error.line}` : `línea ${error.line}, columna ${error.column}`);
  }

  const where = parts.length > 0 ? ` ${parts.join(", ")}` : "";
  return `[${error.code}]${where}: ${error.detail}`;
}

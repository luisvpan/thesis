/**
 * Handle kind types for visual connection validation.
 * Defines what type of data a handle produces or accepts.
 */

export type HandleKind = "rational" | "cpa" | "keyword";

/**
 * Defines what kinds a target handle accepts, with primary (expected) and
 * tolerated (atypical but valid) categories.
 */
export type HandleAcceptance = {
  /** Kinds primarios: match esperado pedagógicamente. */
  primary: HandleKind[];
  /** Kinds tolerados: la conexión se crea pero se marca como atípica. */
  tolerated?: HandleKind[];
};

export type ConnectionMatch = "compatible" | "tolerated" | "incompatible";

/**
 * Checks the connection match level between a source kind and target acceptance.
 */
export function checkConnection(
  sourceKind: HandleKind,
  targetAcceptance: HandleAcceptance
): ConnectionMatch {
  if (targetAcceptance.primary.includes(sourceKind)) return "compatible";
  if (targetAcceptance.tolerated?.includes(sourceKind)) return "tolerated";
  return "incompatible";
}

/**
 * Determines the visual shape of a handle based on its primary kinds.
 * - Single rational → circle
 * - Single cpa → square
 * - Single keyword (or only keyword) → pill
 * - Multiple kinds (cpa + rational) → rounded-square
 */
export function handleShape(
  primary: HandleKind[]
): "circle" | "square" | "rounded-square" | "pill" {
  // Filter keyword from calculation if there are other kinds
  const effective = primary.filter((k) => k !== "keyword");
  if (effective.length === 0) {
    // Only keyword
    return "pill";
  }
  if (effective.length === 1) {
    return effective[0] === "rational" ? "circle" : "square";
  }
  // Multiple kinds (cpa + rational) → rounded-square
  return "rounded-square";
}

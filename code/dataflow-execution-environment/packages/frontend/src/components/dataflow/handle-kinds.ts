/**
 * Handle kind types for visual connection validation.
 * Defines what type of data a handle produces or accepts.
 */

export type HandleKind = "rational" | "cpa" | "keyword" | "group" | "any";

/**
 * Checks if a source handle can connect to a target handle based on their kinds.
 *
 * @param sourceKind - The kind of data the source handle produces
 * @param targetAccepts - Array of kinds the target handle accepts
 * @returns true if the connection is valid
 */
export function acceptsConnection(
  sourceKind: HandleKind,
  targetAccepts: HandleKind[]
): boolean {
  // "any" source can connect to anything
  if (sourceKind === "any") return true;
  // Target that accepts "any" can receive anything
  if (targetAccepts.includes("any")) return true;
  // Check if target accepts this specific kind
  return targetAccepts.includes(sourceKind);
}

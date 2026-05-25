import type Fraction from "fraction.js";

const isFraction = (obj: unknown): obj is Fraction =>
  obj !== null &&
  typeof obj === "object" &&
  "s" in obj &&
  "d" in obj &&
  "n" in obj &&
  typeof (obj as Fraction).toFraction === "function";

/**
 * JSON replacer: Fraction.js → fraction string; BigInt → decimal string.
 */
export function jsonReplacer(_key: string, value: unknown): unknown {
  if (isFraction(value)) {
    return value.d === 1n ? String(value.n * value.s) : value.toFraction();
  }
  if (typeof value === "bigint") {
    return String(value);
  }
  return value;
}

/** Deep-clone value with BigInt/Fraction fields converted to JSON-safe primitives. */
export function toJsonSafe<T>(value: T): T {
  if (value === null || value === undefined) return value;
  if (typeof value === "bigint") return String(value) as T;
  if (isFraction(value)) return jsonReplacer("", value) as T;
  if (typeof value !== "object") return value;

  try {
    return JSON.parse(JSON.stringify(value, jsonReplacer)) as T;
  } catch {
    return value;
  }
}

/** JSON.stringify that never throws on BigInt/Fraction values. */
export function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value, jsonReplacer);
  } catch {
    return "";
  }
}

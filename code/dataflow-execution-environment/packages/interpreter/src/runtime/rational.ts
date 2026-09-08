import Fraction from "fraction.js";
import { RuntimeError } from "./errors";

/**
 * Converts a JS number or string to a Fraction.
 * CRITICAL: Never use native JS operators (+, -, *, /) on numeric values.
 * Always use Fraction.js methods for all arithmetic.
 */
export function toFraction(value: number | string): Fraction {
  return new Fraction(value);
}

export function fromFraction(frac: Fraction): number {
  return frac.valueOf();
}

// Arithmetic operations
export function add(a: Fraction, b: Fraction): Fraction {
  return a.add(b);
}

export function subtract(a: Fraction, b: Fraction): Fraction {
  return a.sub(b);
}

export function multiply(a: Fraction, b: Fraction): Fraction {
  return a.mul(b);
}

export function divide(a: Fraction, b: Fraction): Fraction {
  if (b.valueOf() === 0) {
    throw new RuntimeError("DIVISION_BY_ZERO", "Cannot divide by zero");
  }
  return a.div(b);
}

// Comparison operations
export function lessThan(a: Fraction, b: Fraction): boolean {
  return a.compare(b) < 0;
}

export function greaterThan(a: Fraction, b: Fraction): boolean {
  return a.compare(b) > 0;
}

export function equals(a: Fraction, b: Fraction): boolean {
  return a.equals(b);
}

export function compare(a: Fraction, b: Fraction): number {
  return a.compare(b);
}

// Identity elements
export function zero(): Fraction {
  return new Fraction(0);
}

export function one(): Fraction {
  return new Fraction(1);
}

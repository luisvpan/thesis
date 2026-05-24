/**
 * Pino-based logger with Fraction.js and BigInt serialization support.
 *
 * Usage:
 *   import { logger } from '@/lib/logger';
 *   logger.execute.info("Processing results", { size: 3, fraction: someFraction });
 */

import pino from "pino";
import type Fraction from "fraction.js";

// ---------------------------------------------------------------------------
// Fraction.js / BigInt Serialization
// ---------------------------------------------------------------------------

/**
 * Duck-type check for Fraction.js instances.
 * Checks for internal properties (s, d, n) and the toFraction method.
 */
const isFraction = (obj: unknown): obj is Fraction =>
  obj !== null &&
  typeof obj === "object" &&
  "s" in obj &&
  "d" in obj &&
  "n" in obj &&
  typeof (obj as Fraction).toFraction === "function";

/**
 * JSON replacer that handles Fraction.js and BigInt values.
 * - Fraction.js: converts to "3/4" or "3" if integer
 * - BigInt: converts to "123n" string
 */
const replacer = (_key: string, value: unknown): unknown => {
  if (isFraction(value)) {
    // If denominator is 1, it's an integer - show as number string
    // Otherwise show as fraction string like "3/4"
    return value.d === 1n ? String(value.n * value.s) : value.toFraction();
  }
  if (typeof value === "bigint") {
    return `${value}n`;
  }
  return value;
};

/**
 * Pre-serialize objects containing Fraction.js or BigInt before passing to Pino.
 * This avoids Pino's internal serializer throwing on BigInt.
 */
const serialize = (obj: unknown): unknown => {
  if (obj === null || obj === undefined) return obj;
  if (isFraction(obj)) return replacer("", obj);
  if (typeof obj === "bigint") return `${obj}n`;
  if (typeof obj !== "object") return obj;

  try {
    return JSON.parse(JSON.stringify(obj, replacer));
  } catch {
    // If serialization fails, return a safe representation
    return { _serializationError: true, type: typeof obj };
  }
};

// ---------------------------------------------------------------------------
// Pino Configuration
// ---------------------------------------------------------------------------

// Log level based on environment: debug in dev, info in production
const level = import.meta.env.DEV ? "debug" : "info";

// Base Pino logger for browser
const baseLogger = pino({
  browser: {
    asObject: true,
    formatters: {
      level: (label) => ({ level: label }),
    },
  },
  level,
});

// ---------------------------------------------------------------------------
// Logger Factory
// ---------------------------------------------------------------------------

export type LogFn = (msg: string, obj?: object) => void;

export interface Logger {
  debug: LogFn;
  info: LogFn;
  warn: LogFn;
  error: LogFn;
}

/**
 * Creates a logger instance with a specific context.
 * The context appears in each log entry for filtering.
 */
const createLogger = (context: string): Logger => {
  const child = baseLogger.child({ context });

  return {
    debug: (msg: string, obj?: object) =>
      child.debug(obj ? serialize(obj) : {}, msg),
    info: (msg: string, obj?: object) =>
      child.info(obj ? serialize(obj) : {}, msg),
    warn: (msg: string, obj?: object) =>
      child.warn(obj ? serialize(obj) : {}, msg),
    error: (msg: string, obj?: object) =>
      child.error(obj ? serialize(obj) : {}, msg),
  };
};

// ---------------------------------------------------------------------------
// Pre-configured Loggers (matching existing [prefix] conventions)
// ---------------------------------------------------------------------------

export const logger = {
  /** Program execution flow: [execute] */
  execute: createLogger("execute"),

  /** Frontend to interpreter communication: [frontend→interpreter] */
  interpreter: createLogger("frontend→interpreter"),

  /** Touch system: [touch] */
  touch: createLogger("touch"),

  /** Vision system: [ide:vision] */
  vision: createLogger("ide:vision"),

  /** Flow graph conversion: [flowToProgram] */
  flow: createLogger("flowToProgram"),

  /** Node provider lifecycle: [NodeProvider] */
  nodeProvider: createLogger("NodeProvider"),

  /** Program executor: [executeProgram] */
  executeProgram: createLogger("executeProgram"),

  /** Backend API responses: [backend:elysia] */
  backendElysia: createLogger("backend:elysia"),
};

export { createLogger };

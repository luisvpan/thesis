/**
 * Pino-based logger with Fraction.js and BigInt serialization support.
 *
 * Usage:
 *   import { logger } from '@/lib/logger';
 *   logger.execute.info("Processing results", { size: 3, fraction: someFraction });
 */

import pino from "pino";
import { toJsonSafe } from "@/utils/jsonReplacer";

/** Pre-serialize objects containing Fraction.js or BigInt before passing to Pino. */
const serialize = (obj: unknown): unknown => toJsonSafe(obj);

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

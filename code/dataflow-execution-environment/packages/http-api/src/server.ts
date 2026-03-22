import { Elysia, t } from "elysia";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";
import type { DataflowProgram, ValidationResult } from "@dataflow/shared/types";
import { DagValidator } from "@dataflow/compiler";
import { createRateLimiter } from "@dataflow/shared/security/rate-limiter";
import {
  CompileRequestSchema,
  ExecuteRequestSchema
} from "./schemas";

/**
 * Calculates the maximum depth of a dataflow program's DAG structure
 * @param {DataflowProgram} program - The dataflow program to analyze
 * @returns {number} The maximum depth from any root node to a leaf node
 * @throws {Error} If program structure is invalid
 * @example
 * const depth = calculateMaxDepth(program);
 * console.log(`Program depth: ${depth}`);
 */
function calculateMaxDepth(program: DataflowProgram): number {
  if (!program?.graph?.nodes || program.graph.nodes.length === 0) {
    return 0;
  }

  const nodeIds = new Set<string>(program.graph.nodes.map(n => n.id));
  const edges = program.graph.edges || [];

  function getDepth(nodeId: string, visited: Set<string>): number {
    if (visited.has(nodeId)) {
      return 0;
    }
    visited.add(nodeId);

    const node = program.graph!.nodes.find(n => n.id === nodeId);
    if (!node) {
      return 0;
    }

    let maxChildDepth = 0;
    if (node.type === "DataSource" || node.type === "Output") {
      return 0;
    }

    if (node.inputs) {
      for (const inputId of node.inputs) {
        if (typeof inputId === "string") {
          const childDepth = getDepth(inputId, new Set(visited));
          maxChildDepth = Math.max(maxChildDepth, childDepth);
        }
      }
    }

    return 1 + maxChildDepth;
  }

  let maxDepth = 0;
  for (const nodeId of nodeIds) {
    const depth = getDepth(nodeId, new Set());
    maxDepth = Math.max(maxDepth, depth);
  }

  return maxDepth;
}

const startTime = Date.now();
const validator = new DagValidator();

/**
 * Maximum number of nodes allowed in a dataflow program
 * @constant {number}
 */
const MAX_NODES = 1000;

/**
 * Maximum allowed depth of a dataflow program's DAG structure
 * @constant {number}
 */
const MAX_DEPTH = 10;

/**
 * Maximum memory limit in MB for program execution
 * @constant {number}
 */
const MAX_MEMORY_MB = 100;

/**
 * Maximum number of concurrent program executions allowed
 * @constant {number}
 */
const MAX_CONCURRENT_EXECUTIONS = 5;

/**
 * Counter tracking currently active program executions
 * @type {number}
 */
let activeExecutions = 0;

/**
 * Rate limiter for API requests (100 requests per 60 seconds)
 * @type {import("@dataflow/shared/security/rate-limiter").RateLimiter}
 */
const rateLimiter = createRateLimiter({
  windowMs: 60000,
  maxRequests: 100
});

const healthRoutes = new Elysia({ prefix: '/api/v1' })
  /**
   * Health check endpoint
   * @route GET /api/v1/health
   * @returns {Object} Health status response
   * @property {string} status - Current health status ("healthy")
   * @property {string} version - API version
   * @property {number} uptime - Server uptime in seconds
   * @example
   * const response = await fetch('/api/v1/health');
   * const { status, version, uptime } = await response.json();
   */
  .get("/health", () => {
    const uptime = Math.floor((Date.now() - startTime) / 1000);
    return {
      status: "healthy" as const,
      version: "1.0.0",
      uptime
    };
  });

const compileRoutes = new Elysia({ prefix: '/api/v1' })
  /**
   * Rate limit middleware for compile endpoint
   * Checks IP-based rate limits before processing requests
   * @param {Object} context - Request context
   * @param {Request} context.request - HTTP request object
   * @throws {Error} If rate limit is exceeded
   */
  .onBeforeHandle(({ request }) => {
    const ip = request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown';
    const check = rateLimiter.check(ip);

    if (!check.allowed) {
      throw new Error(`Rate limit exceeded. Try again after ${Math.ceil((check.resetTime - Date.now()) / 1000)}s`);
    }
  })
  /**
   * Compile a dataflow program without executing it
   * @route POST /api/v1/compile
   * @param {Object} body - Request body
   * @param {DataflowProgram} body.program - The dataflow program to compile
   * @returns {CompileResponse} Compilation result with success status, errors, and warnings
   * @throws {Error} If program is too large, too deep, or rate limit exceeded
   * @example
   * const response = await fetch('/api/v1/compile', {
   *   method: 'POST',
   *   headers: { 'Content-Type': 'application/json' },
   *   body: JSON.stringify({ program: myProgram })
   * });
   * const result = await response.json();
   * console.log(result.success, result.errors);
   */
  .post("/compile", ({ body }) => {
    const { program } = body;
    const programId = program.metadata?.programId || "unknown";

    if (program.graph?.nodes?.length > MAX_NODES) {
      throw new Error(`Program too large: ${program.graph.nodes.length} nodes exceeds maximum of ${MAX_NODES}`);
    }

    const depth = calculateMaxDepth(program);
    if (depth > MAX_DEPTH) {
      throw new Error(`Program too deep: depth ${depth} exceeds maximum of ${MAX_DEPTH} levels`);
    }

    const validationResult = validator.validateProgram(program);

    if (!validationResult.success) {
      return {
        success: false,
        programId,
        errors: validationResult.errors,
        warnings: validationResult.warnings
      };
    }

    return {
      success: true,
      programId,
      errors: validationResult.errors,
      warnings: validationResult.warnings
    };
  }, {
    body: CompileRequestSchema
  });

const executeRoutes = new Elysia({ prefix: '/api/v1' })
  /**
   * Rate limit and concurrency middleware for execute endpoint
   * Checks IP-based rate limits and concurrent execution limits before processing requests
   * @param {Object} context - Request context
   * @param {Request} context.request - HTTP request object
   * @throws {Error} If rate limit is exceeded or concurrent execution limit reached
   */
  .onBeforeHandle(({ request }) => {
    const ip = request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown';
    const check = rateLimiter.check(ip);

    if (!check.allowed) {
      throw new Error(`Rate limit exceeded. Try again after ${Math.ceil((check.resetTime - Date.now()) / 1000)}s`);
    }

    if (activeExecutions >= MAX_CONCURRENT_EXECUTIONS) {
      throw new Error(`Too many concurrent executions: maximum of ${MAX_CONCURRENT_EXECUTIONS} reached`);
    }
  })
  /**
   * Execute a dataflow program and return outputs
   * @route POST /api/v1/execute
   * @param {Object} body - Request body
   * @param {DataflowProgram} body.program - The dataflow program to execute
   * @param {Object} body.options - Optional execution parameters
   * @param {number} [body.options.maxTimesteps=100] - Maximum number of timesteps to execute
   * @param {boolean} [body.options.includeTrace=false] - Whether to include execution trace
   * @param {string} [body.options.traceLevel="medium"] - Level of detail for execution trace
   * @returns {ExecuteResponse} Execution result with outputs and optional trace
   * @throws {Error} If program is too large, too deep, rate limit exceeded, timeout, or validation fails
   * @example
   * const response = await fetch('/api/v1/execute', {
   *   method: 'POST',
   *   headers: { 'Content-Type': 'application/json' },
   *   body: JSON.stringify({ 
   *     program: myProgram,
   *     options: { includeTrace: true }
   *   })
   * });
   * const result = await response.json();
   * console.log(result.outputs, result.trace);
   */
  .post("/execute", async ({ body }) => {
    const { program, options = {} } = body;
    const maxTimesteps = options.maxTimesteps || 100;
    const includeTrace = options.includeTrace || false;
    const traceLevel = options.traceLevel || "medium";

    const programId = program.metadata?.programId || "unknown";

    if (program.graph?.nodes?.length > MAX_NODES) {
      throw new Error(`Program too large: ${program.graph.nodes.length} nodes exceeds maximum of ${MAX_NODES}`);
    }

    const depth = calculateMaxDepth(program);
    if (depth > MAX_DEPTH) {
      throw new Error(`Program too deep: depth ${depth} exceeds maximum of ${MAX_DEPTH} levels`);
    }

    const validationResult = validator.validateProgram(program);

    if (!validationResult.success) {
      return {
        success: false,
        programId,
        errors: validationResult.errors,
        warnings: validationResult.warnings
      };
    }

    activeExecutions++;

    try {
      const runtime = new Runtime();
      runtime.loadProgram(program);

      const execStartTime = performance.now();
      const TIMEOUT_MS = 5000;

      const executionPromise = runtime.execute(0);

      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
          reject(new Error('Execution timeout'));
        }, TIMEOUT_MS);
      });

      const outputs = await Promise.race([executionPromise, timeoutPromise]) as unknown[];
      const totalTime = (performance.now() - execStartTime).toFixed(2) + "ms";

      const executionTrace: any = includeTrace ? {
        ...runtime.getExecutionTrace(),
        cacheHits: 0,
        cacheMisses: 0,
        totalTime
      } : undefined;

      if (includeTrace) {
        const evaluator = runtime.getEvaluator();
        const stats = evaluator.getCacheStats();
        executionTrace!.cacheHits = stats.hits;
        executionTrace!.cacheMisses = stats.misses;
      }

      return {
        success: true,
        programId,
        outputs,
        trace: executionTrace
      };
    } finally {
      activeExecutions--;
    }
  }, {
    body: ExecuteRequestSchema
  });

export const app = new Elysia()
  /**
   * Global error handler for the HTTP API server
   * Maps different error types to appropriate HTTP status codes and responses
   * @param {Object} context - Error context
   * @param {string} context.code - Error code (VALIDATION, PARSE, NOT_FOUND, etc.)
   * @param {Error} context.error - The error object
   * @param {Object} context.set - Response setter object
   * @returns {ErrorResponse} Formatted error response with status code
   * @throws {Error} Unhandled errors are caught and returned with status 500
   */
  .onError(({ code, error, set }) => {
    if (code === 'VALIDATION') {
      set.status = 422;
      return {
        success: false,
        errors: [{
          code: 'VALIDATION_ERROR',
          message: (error as any)?.message || 'Invalid request format',
          path: (error as any)?.path
        }]
      };
    }
    
    if (code === 'PARSE') {
      set.status = 400;
      return {
        success: false,
        error: 'Invalid JSON format'
      };
    }
    
    if (code === 'NOT_FOUND') {
      set.status = 404;
      return {
        success: false,
        error: 'Not found'
      };
    }
    
    if (error instanceof Error && error.message.includes('Rate limit exceeded')) {
      set.status = 429;
      return {
        success: false,
        error: 'Too many requests',
        message: error.message
      };
    }
    
    if (error instanceof Error && (error.message.includes('Program too large') || error.message.includes('Program too deep'))) {
      set.status = 413;
      return {
        success: false,
        error: 'Payload too large',
        message: error.message
      };
    }

    if (error instanceof Error && error.message.includes('Too many concurrent executions')) {
      set.status = 429;
      return {
        success: false,
        error: 'Too many concurrent requests',
        message: error.message
      };
    }
    
    if (error instanceof Error && error.message.includes('Execution timeout')) {
      set.status = 408;
      return {
        success: false,
        error: 'Request timeout',
        message: 'Execution took too long'
      };
    }
    
    set.status = 500;
    return {
      success: false,
      error: 'Internal server error'
    };
  })
  .use(healthRoutes)
  .use(compileRoutes)
  .use(executeRoutes);

export type App = typeof app;

import { Elysia, t } from "elysia";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";
import type { DataflowProgram, ValidationResult } from "@dataflow/shared/types";
import { DagValidator } from "@dataflow/compiler";
import { createRateLimiter } from "@dataflow/shared/security/rate-limiter";

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

const MAX_NODES = 1000;
const MAX_DEPTH = 10;
const MAX_MEMORY_MB = 100;
const MAX_CONCURRENT_EXECUTIONS = 5;

let activeExecutions = 0;

const rateLimiter = createRateLimiter({
  windowMs: 60000,
  maxRequests: 100
});

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

const healthRoutes = new Elysia({ prefix: '/api/v1' })
  .get("/health", () => {
    const uptime = Math.floor((Date.now() - startTime) / 1000);
    return {
      status: "healthy",
      version: "1.0.0",
      uptime
    };
  });

const compileRoutes = new Elysia({ prefix: '/api/v1' })
  .onBeforeHandle(({ request }) => {
    const ip = request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown';
    const check = rateLimiter.check(ip);
    
    if (!check.allowed) {
      throw new Error(`Rate limit exceeded. Try again after ${Math.ceil((check.resetTime - Date.now()) / 1000)}s`);
    }
  })
  .post("/compile", ({ body }) => {
    const request = body as any;
    const program = request.program as DataflowProgram;
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
    body: t.Any()
  });

const executeRoutes = new Elysia({ prefix: '/api/v1' })
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
  .post("/execute", async ({ body }) => {
    const request = body as any;
    const program = request.program as DataflowProgram;
    const options = request.options || {};
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
      
      let timeoutTriggered = false;
      const executionPromise = new Promise((resolve, reject) => {
        setImmediate(() => {
          if (timeoutTriggered) {
            reject(new Error('Execution timeout'));
            return;
          }
          
          try {
            const outputs = runtime.execute(0);
            resolve(outputs);
          } catch (error) {
            reject(error);
          }
        });
      });
      
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
          timeoutTriggered = true;
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
        outputs,
        trace: executionTrace
      };
    } finally {
      activeExecutions--;
    }
  }, {
    body: t.Any()
  });

export const app = new Elysia()
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

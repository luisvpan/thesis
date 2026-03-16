import { Elysia, t } from "elysia";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";
import type { DataflowProgram, ValidationResult } from "@dataflow/shared/types";
import { DagValidator } from "@dataflow/compiler";

const startTime = Date.now();
const validator = new DagValidator();

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
    
    set.status = 500;
    return {
      success: false,
      error: 'Internal server error'
    };
  })
  .get("/api/v1/health", () => {
    const uptime = Math.floor((Date.now() - startTime) / 1000);
    return {
      status: "healthy",
      version: "1.0.0",
      uptime
    };
  })
  .post("/api/v1/compile", ({ body }) => {
    const request = body as any;
    const program = request.program as DataflowProgram;
    const programId = program.metadata?.programId || "unknown";
    
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
  })
  .post("/api/v1/execute", ({ body }) => {
    const request = body as any;
    const program = request.program as DataflowProgram;
    const options = request.options || {};
    const maxTimesteps = options.maxTimesteps || 100;
    const includeTrace = options.includeTrace || false;
    const traceLevel = options.traceLevel || "medium";
    
    const programId = program.metadata?.programId || "unknown";
    
    const validationResult = validator.validateProgram(program);
    
    if (!validationResult.success) {
      return {
        success: false,
        programId,
        errors: validationResult.errors,
        warnings: validationResult.warnings
      };
    }
    
    const runtime = new Runtime();
    runtime.loadProgram(program);

    const execStartTime = performance.now();
    const outputs = runtime.execute(0);
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
  }, {
    body: t.Any()
  });

export type App = typeof app;
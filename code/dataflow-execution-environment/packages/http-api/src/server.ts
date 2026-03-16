import { Elysia } from "elysia";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";
import type { DataflowProgram, ValidationResult } from "@dataflow/shared/types";
import { DagValidator } from "@dataflow/compiler";

interface CompileRequest {
  program: DataflowProgram;
}

interface ExecuteRequest {
  program: DataflowProgram;
  options?: {
    maxTimesteps?: number;
    includeTrace?: boolean;
    traceLevel?: "minimal" | "medium" | "detailed";
  };
}

interface ExecutionTrace {
  executionOrder: string[];
  nodeEvaluations: Record<string, { value: unknown; timestep: number }>;
  cacheHits: number;
  cacheMisses: number;
  totalTime: string;
}

interface HealthResponse {
  status: string;
  version: string;
  uptime: number;
}

const startTime = Date.now();
const validator = new DagValidator();

export const app = new Elysia()
  .get("/api/v1/health", () => {
    const uptime = Math.floor((Date.now() - startTime) / 1000);
    const response: HealthResponse = {
      status: "healthy",
      version: "1.0.0",
      uptime
    };
    return response;
  })
  .post("/api/v1/compile", ({ body }) => {
    const request = body as CompileRequest;
    
    const programId = request.program.metadata?.programId || "unknown";
    
    const validationResult = validator.validateProgram(request.program);
    
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
  })
  .post("/api/v1/execute", ({ body }) => {
    const request = body as ExecuteRequest;
    
    const options = request.options || {};
    const maxTimesteps = options.maxTimesteps || 100;
    const includeTrace = options.includeTrace || false;
    const traceLevel = options.traceLevel || "medium";
    
    const programId = request.program.metadata?.programId || "unknown";
    
    const validationResult = validator.validateProgram(request.program);
    
    if (!validationResult.success) {
      return {
        success: false,
        programId,
        errors: validationResult.errors,
        warnings: validationResult.warnings
      };
    }
    
    const runtime = new Runtime();
    runtime.loadProgram(request.program);

    const startTime = performance.now();
    const outputs = runtime.execute(0);
    const totalTime = (performance.now() - startTime).toFixed(2) + "ms";

    const executionTrace: ExecutionTrace | undefined = includeTrace ? {
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
  });

export type App = typeof app;
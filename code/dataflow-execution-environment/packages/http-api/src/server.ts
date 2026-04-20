import { Elysia, t } from "elysia";
import { Compiler, programToSource } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";
import type { DataflowProgram, ValidationResult } from "@dataflow/shared/types";
import { DagValidator } from "@dataflow/compiler";
import { createRateLimiter } from "@dataflow/shared/security/rate-limiter";
import { visionModule } from "./vision";
import { touchModule } from "./touch";

const startTime = Date.now();
const validator = new DagValidator();
const compiler = new Compiler();

const rateLimiter = createRateLimiter({
  windowMs: 60000,
  maxRequests: 100
});

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
  })
  .post("/execute", async ({ body }) => {
    const request = body as any;
    const program = request.program as DataflowProgram;
    const options = request.options || {};
    const maxTimesteps = options.maxTimesteps || 100;
    const includeTrace = options.includeTrace || false;
    const traceLevel = options.traceLevel || "medium";

    const programId = program.metadata?.programId || "unknown";

    // Convertir DataflowProgram a source DSL y compilar para validación completa
    const sourceCode = programToSource(program);
    console.log("[execute] Source DSL generado:\n", sourceCode);
    const compilationResult = compiler.compile(sourceCode);

    if (!compilationResult.success) {
      return {
        success: false,
        programId,
        errors: compilationResult.errors,
        warnings: compilationResult.warnings
      };
    }

    //TODO: adjust typing so if compilationResult.success is true, then compilationResult.program is guaranteed to be defined
    const runtime = new Runtime();
    runtime.loadProgram(compilationResult.program!);

    const execStartTime = performance.now();
    const TIMEOUT_MS = 5000;

    const executionPromise = new Promise((resolve) => {
      const outputs = runtime.execute(0);
      resolve(outputs);
    });

    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Execution timeout')), TIMEOUT_MS);
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
  .use(executeRoutes)
  .use(visionModule)
  .use(touchModule);

export type App = typeof app;
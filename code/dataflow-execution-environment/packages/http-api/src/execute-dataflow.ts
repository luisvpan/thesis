import { Compiler, programToSource } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";
import type { DataflowProgram } from "@dataflow/shared/types";

const compiler = new Compiler();

export type ExecuteDataflowOk = {
  success: true;
  programId: string;
  outputs: unknown[];
  totalTimeMs: string;
};

export type ExecuteDataflowErr = {
  success: false;
  programId: string;
  errors: unknown;
};

export type ExecuteDataflowResult = ExecuteDataflowOk | ExecuteDataflowErr;

/**
 * Compila y ejecuta un programa dataflow (misma lógica que POST /api/v1/execute).
 */
export function executeDataflowProgram(program: DataflowProgram): ExecuteDataflowResult {
  const programId = program.metadata?.programId || "unknown";
  const sourceCode = programToSource(program);
  console.log("[execute-dataflow] DSL:\n", sourceCode);
  const compilationResult = compiler.compile(sourceCode);

  if (!compilationResult.success) {
    return {
      success: false,
      programId,
      errors: compilationResult.errors,
    };
  }

  const execStart = performance.now();
  const runtime = new Runtime();
  runtime.loadProgram(compilationResult.program!);
  const outputs = runtime.execute(0);
  const totalTimeMs = (performance.now() - execStart).toFixed(2) + "ms";

  return {
    success: true,
    programId,
    outputs,
    totalTimeMs,
  };
}

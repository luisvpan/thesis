import { describe } from 'bun:test';
import { Runtime } from '../runtime';
import { IncrementalRuntime } from '../incremental-runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

export type RuntimeInstance = Runtime | IncrementalRuntime;

export interface RuntimeTestContext {
  runtime: RuntimeInstance;
  loadProgram: (program: DataflowProgram) => void;
  execute: (time?: number) => Promise<unknown[] | { nodeStates: Map<string, any>; changedNodes: string[] }>;
  getOutput: (nodeId: string, time?: number) => Promise<any>;
  getCacheStats: () => { hits: number; misses: number };
}

export function createRuntimeContext(runtimeType: 'batch' | 'incremental'): RuntimeTestContext {
  const runtime = runtimeType === 'batch' ? new Runtime() : new IncrementalRuntime();
  
  return {
    runtime,
    loadProgram: (program: DataflowProgram) => {
      runtime.loadProgram(program);
    },
    execute: async (time?: number) => {
      if (runtimeType === 'batch') {
        return await (runtime as Runtime).execute(time);
      } else {
        return await (runtime as IncrementalRuntime).evaluatePartial(time || 0);
      }
    },
    getOutput: async (nodeId: string, time?: number) => {
      if (runtimeType === 'batch') {
        const outputs = await (runtime as Runtime).execute(time);
        const evaluator = (runtime as Runtime).getEvaluator();
        return await evaluator.evaluate(nodeId, time || 0, (runtime as Runtime).getGraph());
      } else {
        const evalResult = await (runtime as IncrementalRuntime).evaluatePartial(time || 0);
        const state = evalResult.nodeStates.get(nodeId);
        return state && state.status === 'completed' ? state.value : null;
      }
    },
    getCacheStats: () => {
      return runtime.getCacheStats();
    }
  };
}

export function describeWithBothRuntimes(name: string, testFn: (context: RuntimeTestContext) => void) {
  describe(`${name} (batch)`, () => {
    const context = createRuntimeContext('batch');
    testFn(context);
  });

  describe(`${name} (incremental)`, () => {
    const context = createRuntimeContext('incremental');
    testFn(context);
  });
}

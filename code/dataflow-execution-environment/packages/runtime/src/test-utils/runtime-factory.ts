import { describe } from 'bun:test';
import { Runtime } from '../runtime';
import { IncrementalRuntime } from '../incremental-runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

export type RuntimeInstance = Runtime | IncrementalRuntime;

export interface RuntimeTestContext {
  runtime: RuntimeInstance;
  loadProgram: (program: DataflowProgram) => void;
  execute: (time?: number) => unknown[] | { nodeStates: Map<string, any>; changedNodes: string[] };
  getOutput: (nodeId: string, time?: number) => any;
}

export function createRuntimeContext(runtimeType: 'batch' | 'incremental'): RuntimeTestContext {
  const runtime = runtimeType === 'batch' ? new Runtime() : new IncrementalRuntime();

  return {
    runtime,
    loadProgram: (program: DataflowProgram) => {
      runtime.loadProgram(program);
    },
    execute: (time?: number) => {
      if (runtimeType === 'batch') {
        return (runtime as Runtime).execute(time);
      } else {
        return (runtime as IncrementalRuntime).evaluatePartial(time || 0);
      }
    },
    getOutput: (nodeId: string, time?: number) => {
      if (runtimeType === 'batch') {
        const outputs = (runtime as Runtime).execute(time);
        return outputs[0];
      } else {
        const evalResult = (runtime as IncrementalRuntime).evaluatePartial(time || 0);
        const state = evalResult.nodeStates.get(nodeId);
        return state && state.status === 'completed' ? state.value : null;
      }
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

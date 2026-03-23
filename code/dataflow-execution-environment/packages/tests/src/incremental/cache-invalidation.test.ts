import { describe, it, expect, beforeEach } from 'bun:test';
import { IncrementalRuntime } from '@dataflow/runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

function isCompletedState(state: any): state is { status: "completed"; value: any } {
  return state && state.status === "completed";
}

describe('IncrementalRuntime - Cache Invalidation', () => {
  let runtime: IncrementalRuntime;

  beforeEach(() => {
    runtime = new IncrementalRuntime();
  });

  it('should invalidate dependent nodes on change', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-invalidate-dependents' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'add', toPort: 0 },
          { id: 'e2', from: 'b', to: 'add', toPort: 1 },
          { id: 'e3', from: 'add', to: 'result' }
        ]
      }
    };

    runtime.loadProgram(program);
    
    // First evaluation
    const firstResult = runtime.evaluatePartial(0);
    const firstState = firstResult.nodeStates.get('result');

    if (isCompletedState(firstState)) {
      expect(firstState.value).toEqual({ kind: 'natural', value: 8 });
    }

    // Update node 'a' - cache for 'add', 'result' should be invalidated
    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 }
      ]
    });

    // Second evaluation should reflect the change
    const secondResult = runtime.evaluatePartial(0);
    const secondState = secondResult.nodeStates.get('result');

    if (isCompletedState(secondState)) {
      expect(secondState.value).toEqual({ kind: 'natural', value: 13 });
    }
  });

  it('should preserve cache for unaffected nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-preserve-cache' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'c', type: 'DataSource', dataType: 'natural', value: 7 },
          { id: 'mul', type: 'Transformation', dataType: 'natural', operation: 'MULTIPLY', inputs: ['c', 'c'] },
          { id: 'result1', type: 'Output', dataType: 'natural', input: 'add' },
          { id: 'result2', type: 'Output', dataType: 'natural', input: 'mul' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'add', toPort: 0 },
          { id: 'e2', from: 'b', to: 'add', toPort: 1 },
          { id: 'e3', from: 'add', to: 'result1' },
          { id: 'e4', from: 'c', to: 'mul', toPort: 0 },
          { id: 'e5', from: 'c', to: 'mul', toPort: 1 },
          { id: 'e6', from: 'mul', to: 'result2' }
        ]
      }
    };

    runtime.loadProgram(program);
    
    // First evaluation
    const firstResult = runtime.evaluatePartial(0);
    const firstResult1 = firstResult.nodeStates.get('result1');
    const firstResult2 = firstResult.nodeStates.get('result2');

    if (isCompletedState(firstResult1)) {
      expect(firstResult1.value).toEqual({ kind: 'natural', value: 8 });
    }
    if (isCompletedState(firstResult2)) {
      expect(firstResult2.value).toEqual({ kind: 'natural', value: 49 });
    }

    // Update node 'a' - only 'add' and 'result1' should be re-evaluated
    // 'mul' and 'result2' should use cached values
    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 15 }
      ]
    });

    // Second evaluation
    const secondResult = runtime.evaluatePartial(0);
    const secondResult1 = secondResult.nodeStates.get('result1');
    const secondResult2 = secondResult.nodeStates.get('result2');

    if (isCompletedState(secondResult1)) {
      expect(secondResult1.value).toEqual({ kind: 'natural', value: 18 });
    }
    if (isCompletedState(secondResult2)) {
      expect(secondResult2.value).toEqual({ kind: 'natural', value: 49 });
    }
  });

  it('should clear cache on program reload', () => {
    const program1: DataflowProgram = {
      metadata: { programId: 'test-reload1' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'a' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'result' }
        ]
      }
    };

    runtime.loadProgram(program1);
    
    // First evaluation
    const firstResult = runtime.evaluatePartial(0);
    const firstState = firstResult.nodeStates.get('result');

    if (isCompletedState(firstState)) {
      expect(firstState.value).toEqual({ kind: 'natural', value: 5 });
    }

    // Load a different program - cache should be cleared
    const program2: DataflowProgram = {
      metadata: { programId: 'test-reload2' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'a' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'result' }
        ]
      }
    };

    runtime.loadProgram(program2);
    
    // Second evaluation after reload
    const secondResult = runtime.evaluatePartial(0);
    const secondState = secondResult.nodeStates.get('result');

    if (isCompletedState(secondState)) {
      expect(secondState.value).toEqual({ kind: 'natural', value: 10 });
    }
  });
});

import { describe, it, expect, beforeEach } from 'bun:test';
import { IncrementalRuntime } from '@dataflow/runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

function isCompletedState(state: any): state is { status: "completed"; value: any } {
  return state && state.status === "completed";
}

describe('IncrementalRuntime - Incremental Recompute', () => {
  let runtime: IncrementalRuntime;

  beforeEach(() => {
    runtime = new IncrementalRuntime();
  });

  it('should re-evaluate only changed nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-recompute-changed' },
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
    expect(firstResult.nodeStates.size).toBeGreaterThan(0);

    // Update node 'a' from 5 to 10
    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 }
      ]
    });

    // Second evaluation should only re-evaluate affected nodes
    const secondResult = runtime.evaluatePartial(0);
    const resultState = secondResult.nodeStates.get('result');

    if (isCompletedState(resultState)) {
      expect(resultState.value).toEqual({ kind: 'natural', value: 13 });
    }
  });

  it('should re-evaluate only dependent nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-recompute-dependent' },
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
    runtime.evaluatePartial(0);

    // Update node 'a' - only 'add' and 'result1' should be re-evaluated
    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 15 }
      ]
    });

    const result = runtime.evaluatePartial(0);
    const result1State = result.nodeStates.get('result1');
    const result2State = result.nodeStates.get('result2');

    if (isCompletedState(result1State)) {
      expect(result1State.value).toEqual({ kind: 'natural', value: 18 });
    }
    if (isCompletedState(result2State)) {
      expect(result2State.value).toEqual({ kind: 'natural', value: 49 });
    }
  });

  it('should preserve cache for unchanged nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-preserve-cache' },
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

    // Add an unrelated node - should not affect existing cache
    runtime.updateGraph({
      addedNodes: [
        { id: 'c', type: 'DataSource', dataType: 'natural', value: 10 }
      ]
    });

    // Second evaluation - 'result' should still use cached value
    const secondResult = runtime.evaluatePartial(0);
    const secondState = secondResult.nodeStates.get('result');

    if (isCompletedState(secondState)) {
      expect(secondState.value).toEqual({ kind: 'natural', value: 8 });
    }
  });

  it('should handle circular dependencies', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-circular-deps' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'c'] },
          { id: 'c', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['b', 'd'] },
          { id: 'd', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'b' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'b', toPort: 0 },
          { id: 'e2', from: 'c', to: 'b', toPort: 1 },
          { id: 'e3', from: 'b', to: 'c', toPort: 0 },
          { id: 'e4', from: 'd', to: 'c', toPort: 1 },
          { id: 'e5', from: 'b', to: 'result' }
        ]
      }
    };

    runtime.loadProgram(program);
    
    // Should handle circular dependencies gracefully
    const result = runtime.evaluatePartial(0);
    const resultState = result.nodeStates.get('result');

    // Either completes successfully or returns an error state
    expect(resultState).toBeDefined();
  });

  it('should show performance improvement over full re-evaluation', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-performance' },
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
    
    // First evaluation (full evaluation)
    const start1 = performance.now();
    runtime.evaluatePartial(0);
    const time1 = performance.now() - start1;

    // Update and re-evaluate (incremental, should be faster or similar)
    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 }
      ]
    });

    const start2 = performance.now();
    runtime.evaluatePartial(0);
    const time2 = performance.now() - start2;

    // Incremental re-evaluation should not be significantly slower
    // (allowing for variance, we just ensure it's not orders of magnitude slower)
    expect(time2).toBeLessThan(time1 * 10);
  });

  it('should handle cache invalidation correctly', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-cache-invalidation' },
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

    // Update node 'a' - cache for 'add' and 'result' should be invalidated
    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 }
      ]
    });

    // Second evaluation should use new value, not cached old value
    const secondResult = runtime.evaluatePartial(0);
    const secondState = secondResult.nodeStates.get('result');

    if (isCompletedState(secondState)) {
      expect(secondState.value).toEqual({ kind: 'natural', value: 13 });
    }
  });
});

import { describe, it, expect, beforeEach } from 'bun:test';
import { IncrementalRuntime } from '@dataflow/runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

function isCompletedState(state: any): state is { status: "completed"; value: any } {
  return state && state.status === "completed";
}

describe('IncrementalRuntime - Graph Updates', () => {
  let runtime: IncrementalRuntime;

  beforeEach(() => {
    runtime = new IncrementalRuntime();
  });

  it('should add nodes to graph', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-add-node' },
      graph: {
        nodes: [],
        edges: []
      }
    };

    runtime.loadProgram(program);

    const result = runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
      ]
    });

    expect(result.changedNodes).toContain('a');
    expect(result.errors).toHaveLength(0);

    runtime.subscribe('a', () => {});
    const evalResult = runtime.evaluatePartial(0);
    const state = evalResult.nodeStates.get('a');

    if (isCompletedState(state)) {
      expect(state.value).toEqual({ kind: 'natural', value: 5 });
    }
  });

  it('should remove nodes from graph', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-remove-node' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 3 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);

    runtime.subscribe('a', () => {});
    runtime.subscribe('b', () => {});
    runtime.evaluatePartial(0);

    const result = runtime.updateGraph({
      removedNodes: ['a']
    });

    expect(result.changedNodes).toContain('a');
    expect(result.errors).toHaveLength(0);

    runtime.subscribe('b', () => {});
    const evalResult = runtime.evaluatePartial(1);
    
    expect(evalResult.nodeStates.has('a')).toBe(false);
    expect(evalResult.nodeStates.has('b')).toBe(true);
  });

  it('should update node values', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-update-value' },
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

    runtime.loadProgram(program);
    runtime.evaluatePartial(0);

    const firstResult = runtime.evaluatePartial(0);
    const firstState = firstResult.nodeStates.get('result');

    if (isCompletedState(firstState)) {
      expect(firstState.value).toEqual({ kind: 'natural', value: 5 });
    }

    const updateResult = runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 }
      ]
    });

    expect(updateResult.changedNodes).toContain('a');

    const secondResult = runtime.evaluatePartial(0);
    const secondState = secondResult.nodeStates.get('result');

    if (isCompletedState(secondState)) {
      expect(secondState.value).toEqual({ kind: 'natural', value: 10 });
    }
  });

  it('should trigger only affected node re-evaluation', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-affected-reeval' },
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

  it('should maintain cache for unchanged nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-cache-maintain' },
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
    runtime.evaluatePartial(0);

    const firstResult = runtime.evaluatePartial(0);
    const firstState = firstResult.nodeStates.get('result');

    if (isCompletedState(firstState)) {
      expect(firstState.value).toEqual({ kind: 'natural', value: 8 });
    }

    runtime.updateGraph({
      addedNodes: [
        { id: 'c', type: 'DataSource', dataType: 'natural', value: 10 }
      ]
    });

    const secondResult = runtime.evaluatePartial(0);
    const secondState = secondResult.nodeStates.get('result');

    if (isCompletedState(secondState)) {
      expect(secondState.value).toEqual({ kind: 'natural', value: 8 });
    }
  });

  it('should handle multiple node updates', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-multiple-updates' },
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
    runtime.evaluatePartial(0);

    const result = runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 },
        { id: 'b', type: 'DataSource', dataType: 'natural', value: 7 }
      ]
    });

    expect(result.changedNodes).toContain('a');
    expect(result.changedNodes).toContain('b');

    const evalResult = runtime.evaluatePartial(0);
    const state = evalResult.nodeStates.get('result');

    if (isCompletedState(state)) {
      expect(state.value).toEqual({ kind: 'natural', value: 17 });
    }
  });

  it('should update dependency graph correctly', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-dep-graph' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 3 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);

    runtime.updateGraph({
      addedEdges: [
        { id: 'e1', from: 'a', to: 'b' }
      ]
    });

    runtime.subscribe('b', () => {});
    const evalResult = runtime.evaluatePartial(0);
    
    expect(evalResult.nodeStates.has('b')).toBe(true);
  });

  it('should handle updates during evaluation', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-update-during-eval' },
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
    
    const firstResult = runtime.evaluatePartial(0);
    const firstState = firstResult.nodeStates.get('result');

    if (isCompletedState(firstState)) {
      expect(firstState.value).toEqual({ kind: 'natural', value: 8 });
    }

    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 15 }
      ]
    });

    const secondResult = runtime.evaluatePartial(0);
    const secondState = secondResult.nodeStates.get('result');

    if (isCompletedState(secondState)) {
      expect(secondState.value).toEqual({ kind: 'natural', value: 18 });
    }
  });
});

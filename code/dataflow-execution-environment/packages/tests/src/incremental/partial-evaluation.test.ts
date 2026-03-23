import { describe, it, expect, beforeEach } from 'bun:test';
import { IncrementalRuntime } from '@dataflow/runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

function isCompletedState(state: any): state is { status: "completed"; value: any } {
  return state && state.status === "completed";
}

function isPendingState(state: any): state is { status: "pending"; missingInputs: any[] } {
  return state && state.status === "pending";
}

function isErrorState(state: any): state is { status: "error"; error: string } {
  return state && state.status === "error";
}

describe('IncrementalRuntime - Partial Evaluation', () => {
  let runtime: IncrementalRuntime;

  beforeEach(() => {
    runtime = new IncrementalRuntime();
  });

  it('should return empty result with no demand sources', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-no-demand' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    const result = runtime.evaluatePartial(0);

    expect(result.nodeStates).toBeInstanceOf(Map);
    expect(result.nodeStates.size).toBe(0);
    expect(result.changedNodes).toHaveLength(0);
  });

  it('should evaluate subscribed nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-subscribed' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    runtime.subscribe('a', () => {});
    
    const result = runtime.evaluatePartial(0);

    expect(result.nodeStates.get('a')).toEqual({
      status: 'completed',
      value: { kind: 'natural', value: 5 }
    });
  });

  it('should evaluate output nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-output' },
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
    const result = runtime.evaluatePartial(0);

    expect(result.nodeStates.get('result')).toEqual({
      status: 'completed',
      value: { kind: 'natural', value: 5 }
    });
  });

  it('should evaluate both outputs and subscriptions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-both' },
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
    runtime.subscribe('a', () => {});
    
    const result = runtime.evaluatePartial(0);

    expect(result.nodeStates.has('a')).toBe(true);
    expect(result.nodeStates.has('result')).toBe(true);
  });

  it('should return completed state for successfully evaluated nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-completed' },
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
    const result = runtime.evaluatePartial(0);

    const outputState = result.nodeStates.get('result');
    if (isCompletedState(outputState)) {
      expect(outputState.status).toBe('completed');
      expect(outputState.value).toEqual({ kind: 'natural', value: 5 });
    }
  });

  it('should return pending state for nodes with missing inputs', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-pending' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'add', toPort: 0 }
        ]
      }
    };

    runtime.loadProgram(program);
    const result = runtime.evaluatePartial(0);

    const addState = result.nodeStates.get('add');
    if (isPendingState(addState)) {
      expect(addState.status).toBe('pending');
      expect(addState.missingInputs).toBeDefined();
      expect(addState.missingInputs).toHaveLength(1);
    }
  });

  it('should return error state for evaluation errors', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-error' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 0 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 0 },
          { id: 'div', type: 'Transformation', dataType: 'decimal', operation: 'DIVIDE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'decimal', input: 'div' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'div', toPort: 0 },
          { id: 'e2', from: 'b', to: 'div', toPort: 1 },
          { id: 'e3', from: 'div', to: 'result' }
        ]
      }
    };

    runtime.loadProgram(program);
    const result = runtime.evaluatePartial(0);

    const resultState = result.nodeStates.get('result');
    if (isErrorState(resultState)) {
      expect(resultState.status).toBe('error');
      expect(resultState.error).toContain('Division by zero');
    }
  });
});

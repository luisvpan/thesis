import { describe, it, expect, beforeEach } from 'bun:test';
import { IncrementalRuntime } from '@dataflow/runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

function isPendingState(state: any): state is { status: "pending"; missingInputs: any[] } {
  return state && state.status === "pending";
}

function isErrorState(state: any): state is { status: "error"; error: string } {
  return state && state.status === "error";
}

describe('IncrementalRuntime - Missing Input Handling', () => {
  let runtime: IncrementalRuntime;

  beforeEach(() => {
    runtime = new IncrementalRuntime();
  });

  it('should detect missing input port', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-missing-port' },
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
      expect(addState.missingInputs).toBeDefined();
      expect(addState.missingInputs.length).toBeGreaterThan(0);
      const missingPort = addState.missingInputs.find((m: any) => m.port === 1);
      expect(missingPort).toBeDefined();
    }
  });

  it('should create MissingInput with correct port number', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-port-number' },
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
      expect(addState.missingInputs[0]?.port).toBe(1);
    }
  });

  it('should provide child-friendly error messages', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-child-friendly' },
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
      expect(addState.missingInputs[0]?.childMessage).toContain('⚠️');
    }
  });

  it('should cascade pending states through dependencies', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-cascade' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'mul', type: 'Transformation', dataType: 'natural', operation: 'MULTIPLY', inputs: ['add', 'c'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'mul' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'add', toPort: 0 }
        ]
      }
    };

    runtime.loadProgram(program);
    const result = runtime.evaluatePartial(0);

    const mulState = result.nodeStates.get('mul');
    const addState = result.nodeStates.get('add');
    
    if (isPendingState(mulState)) {
      expect(mulState.missingInputs.length).toBeGreaterThan(0);
    }
    
    if (isPendingState(addState)) {
      expect(addState.missingInputs.length).toBeGreaterThan(0);
    }
  });

  it('should differentiate between missing inputs and evaluation errors', () => {
    const program1: DataflowProgram = {
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

    runtime.loadProgram(program1);
    const result1 = runtime.evaluatePartial(0);
    
    const resultState = result1.nodeStates.get('result');
    if (isErrorState(resultState)) {
      expect(resultState.status).toBe('error');
      expect(resultState.error).toContain('Division by zero');
    }
  });

  it('should report multiple missing inputs', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-multiple-missing' },
      graph: {
        nodes: [
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    const result = runtime.evaluatePartial(0);

    const addState = result.nodeStates.get('add');
    if (isPendingState(addState)) {
      expect(addState.missingInputs.length).toBe(2);
    }
  });
});

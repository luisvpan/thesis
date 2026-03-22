import { describe, it, expect } from 'bun:test';

import { Runtime } from './runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

describe('Runtime - Program Loading', () => {
  it('should load a simple program', () => {
    const runtime = new Runtime();

    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    expect(() => runtime.loadProgram(program)).not.toThrow();
  });

  it('should load program with multiple nodes', () => {
    const runtime = new Runtime();

    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
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

    expect(() => runtime.loadProgram(program)).not.toThrow();
  });

  it('should replace previous program when loading new one', async () => {
    const runtime = new Runtime();

    const program1: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    const program2: DataflowProgram = {
      metadata: { programId: 'prog_002' },
      graph: {
        nodes: [
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 10 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program1);
    runtime.loadProgram(program2);

    const outputs = await runtime.execute();
    expect(outputs).toHaveLength(0);
  });
});

describe('Runtime - Execution', () => {
  it('should return empty array for program with no output nodes', async () => {
    const runtime = new Runtime();
 
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    const outputs = await runtime.execute();
 
    expect(outputs).toEqual([]);
  });

  it('should execute simple output node', async () => {
    const runtime = new Runtime();
 
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
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
    const outputs = await runtime.execute();
 
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'natural', value: 5 });
  });

  it('should handle error from operation', async () => {
    const runtime = new Runtime();
 
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 },
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
 
    await expect(runtime.execute()).rejects.toThrow('Division by zero');
  });
});

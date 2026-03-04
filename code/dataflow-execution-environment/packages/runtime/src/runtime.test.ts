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

  it('should replace previous program when loading new one', () => {
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
    
    const outputs = runtime.execute();
    expect(outputs).toHaveLength(0);
  });
});

describe('Runtime - Execution', () => {
  it('should return empty array for program with no output nodes', () => {
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
    const outputs = runtime.execute();
    
    expect(outputs).toEqual([]);
  });

  it('should execute simple output node', () => {
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
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'natural', value: 5 });
  });

  it('should execute ADD operation', () => {
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
    
    runtime.loadProgram(program);
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'natural', value: 5 });
  });

  it('should execute SUBTRACT operation', () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'sub', type: 'Transformation', dataType: 'integer', operation: 'SUBTRACT', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'integer', input: 'sub' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'sub', toPort: 0 },
          { id: 'e2', from: 'b', to: 'sub', toPort: 1 },
          { id: 'e3', from: 'sub', to: 'result' }
        ]
      }
    };
    
    runtime.loadProgram(program);
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'integer', value: 7 });
  });

  it('should execute MULTIPLY operation', () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 4 },
          { id: 'mul', type: 'Transformation', dataType: 'natural', operation: 'MULTIPLY', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'mul' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'mul', toPort: 0 },
          { id: 'e2', from: 'b', to: 'mul', toPort: 1 },
          { id: 'e3', from: 'mul', to: 'result' }
        ]
      }
    };
    
    runtime.loadProgram(program);
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'natural', value: 12 });
  });

  it('should execute DIVIDE operation', () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 4 },
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
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'decimal', value: 2.5 });
  });

  it('should throw error for division by zero', () => {
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
    
    expect(() => runtime.execute()).toThrow('Division by zero');
  });

  it('should execute COMPARE operation with equal values', () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'cmp', type: 'Transformation', dataType: 'integer', operation: 'COMPARE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'integer', input: 'cmp' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'cmp', toPort: 0 },
          { id: 'e2', from: 'b', to: 'cmp', toPort: 1 },
          { id: 'e3', from: 'cmp', to: 'result' }
        ]
      }
    };
    
    runtime.loadProgram(program);
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'integer', value: 0 });
  });

  it('should execute COMPARE operation with less than', () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'cmp', type: 'Transformation', dataType: 'integer', operation: 'COMPARE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'integer', input: 'cmp' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'cmp', toPort: 0 },
          { id: 'e2', from: 'b', to: 'cmp', toPort: 1 },
          { id: 'e3', from: 'cmp', to: 'result' }
        ]
      }
    };
    
    runtime.loadProgram(program);
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'integer', value: -1 });
  });

  it('should execute COMPARE operation with greater than', () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 7 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'cmp', type: 'Transformation', dataType: 'integer', operation: 'COMPARE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'integer', input: 'cmp' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'cmp', toPort: 0 },
          { id: 'e2', from: 'b', to: 'cmp', toPort: 1 },
          { id: 'e3', from: 'cmp', to: 'result' }
        ]
      }
    };
    
    runtime.loadProgram(program);
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'integer', value: 1 });
  });

  it('should execute complex expression (3 + 2) * (10 - 6) = 20', () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
          { id: 'c', type: 'DataSource', dataType: 'natural', value: 10 },
          { id: 'd', type: 'DataSource', dataType: 'natural', value: 6 },
          { id: 'sum', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'diff', type: 'Transformation', dataType: 'integer', operation: 'SUBTRACT', inputs: ['c', 'd'] },
          { id: 'product', type: 'Transformation', dataType: 'natural', operation: 'MULTIPLY', inputs: ['sum', 'diff'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'product' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'sum', toPort: 0 },
          { id: 'e2', from: 'b', to: 'sum', toPort: 1 },
          { id: 'e3', from: 'c', to: 'diff', toPort: 0 },
          { id: 'e4', from: 'd', to: 'diff', toPort: 1 },
          { id: 'e5', from: 'sum', to: 'product', toPort: 0 },
          { id: 'e6', from: 'diff', to: 'product', toPort: 1 },
          { id: 'e7', from: 'product', to: 'result' }
        ]
      }
    };
    
    runtime.loadProgram(program);
    const outputs = runtime.execute();
    
    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toEqual({ kind: 'natural', value: 20 });
  });
});

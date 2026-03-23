import type { DataflowProgram } from '@dataflow/shared/types';

export const simpleArithmeticProgram: DataflowProgram = {
  metadata: { programId: 'test-arithmetic' },
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

export const simpleFractionProgram: DataflowProgram = {
  metadata: { programId: 'test-fraction' },
  graph: {
    nodes: [
      { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
      { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 4 } },
      { id: 'add', type: 'Transformation', dataType: 'fraction', operation: 'ADD', inputs: ['a', 'b'] },
      { id: 'result', type: 'Output', dataType: 'fraction', input: 'add' }
    ],
    edges: [
      { id: 'e1', from: 'a', to: 'add', toPort: 0 },
      { id: 'e2', from: 'b', to: 'add', toPort: 1 },
      { id: 'e3', from: 'add', to: 'result' }
    ]
  }
};

export const complexArithmeticProgram: DataflowProgram = {
  metadata: { programId: 'test-complex' },
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

import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes } from '../../../runtime/src/test-utils';
import { expectBoolean, expectNatural } from '../../../runtime/src/test-utils';
import type { DataflowProgram } from '@dataflow/shared/types';

describeWithBothRuntimes('Comparison Operations - COMPARE', (context) => {
  it('should execute COMPARE operation with equal values', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-compare-equal' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'cmp', type: 'Transformation', dataType: 'boolean', operation: 'COMPARE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'boolean', input: 'cmp' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'cmp', toPort: 0 },
          { id: 'e2', from: 'b', to: 'cmp', toPort: 1 },
          { id: 'e3', from: 'cmp', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectBoolean(result, true);
  });

  it('should execute COMPARE operation with less than', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-compare-less' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'cmp', type: 'Transformation', dataType: 'boolean', operation: 'COMPARE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'boolean', input: 'cmp' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'cmp', toPort: 0 },
          { id: 'e2', from: 'b', to: 'cmp', toPort: 1 },
          { id: 'e3', from: 'cmp', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectBoolean(result, false);
  });

  it('should execute COMPARE operation with greater than', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-compare-greater' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 7 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'cmp', type: 'Transformation', dataType: 'boolean', operation: 'COMPARE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'boolean', input: 'cmp' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'cmp', toPort: 0 },
          { id: 'e2', from: 'b', to: 'cmp', toPort: 1 },
          { id: 'e3', from: 'cmp', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectBoolean(result, false);
  });
});

describeWithBothRuntimes('Complex Expressions', (context) => {
  it('should execute complex expression (3 + 2) * (10 - 6) = 20', async () => {
    const program: DataflowProgram = {
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

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectNatural(result, 20);
  });
});

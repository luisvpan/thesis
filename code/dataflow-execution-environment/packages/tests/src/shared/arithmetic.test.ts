import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes } from '../../../runtime/src/test-utils';
import { expectNatural, expectInteger, expectDecimal } from '../../../runtime/src/test-utils';
import type { DataflowProgram } from '@dataflow/shared/types';

describeWithBothRuntimes('Arithmetic Operations - ADD', (context) => {
  it('should execute ADD operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-add' },
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

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectNatural(result, 5);
  });
});

describeWithBothRuntimes('Arithmetic Operations - SUBTRACT', (context) => {
  it('should execute SUBTRACT operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-subtract' },
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

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectInteger(result, 7);
  });
});

describeWithBothRuntimes('Arithmetic Operations - MULTIPLY', (context) => {
  it('should execute MULTIPLY operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-multiply' },
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

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectNatural(result, 12);
  });
});

describeWithBothRuntimes('Arithmetic Operations - DIVIDE', (context) => {
  it('should execute DIVIDE operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-divide' },
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

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectDecimal(result, 2.5);
  });

  it('should handle division by zero error', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-divide-by-zero' },
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

    context.loadProgram(program);

    if (context.runtime.constructor.name === 'Runtime') {
      await expect(context.execute()).rejects.toThrow('Division by zero');
    } else {
      const result = await context.execute();
      const nodeStates = (result as { nodeStates: Map<string, any>; changedNodes: string[] }).nodeStates;
      const errorState = nodeStates.get('result');
      expect(errorState?.status).toBe('error');
      expect(errorState?.error).toContain('Division by zero');
    }
  });
});

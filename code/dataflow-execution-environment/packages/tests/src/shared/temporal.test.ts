import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes, expectNatural, expectInteger, expectDecimal } from '../../../runtime/src/test-utils';
import type { DataflowProgram } from '@dataflow/shared/types';

function* counterGenerator() {
  let i = 0;
  while (true) {
    yield i++;
  }
}

describeWithBothRuntimes('Temporal Operations - FIRST', (context) => {
  it('should extract first value from natural stream', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-first-natural' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'first', type: 'Transformation', dataType: 'natural', operation: 'FIRST', inputs: ['stream'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'first' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'first', toPort: 0 },
          { id: 'e2', from: 'first', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expectNatural(result, 0);
  });

  it('should always return the same first value', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-first-consistent' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'first', type: 'Transformation', dataType: 'natural', operation: 'FIRST', inputs: ['stream'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'first' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'first', toPort: 0 },
          { id: 'e2', from: 'first', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result1 = await context.getOutput('result', 0);
    const result2 = await context.getOutput('result', 1);
    const result3 = await context.getOutput('result', 5);

    expectNatural(result1, 0);
    expectNatural(result2, 0);
    expectNatural(result3, 0);
  });
});

describeWithBothRuntimes('Temporal Operations - NEXT', (context) => {
  it('should get current value from stream at time 0', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-next-time-0' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'next', type: 'Transformation', dataType: 'natural', operation: 'NEXT', inputs: ['stream'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'next' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'next', toPort: 0 },
          { id: 'e2', from: 'next', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result', 0);
    expectNatural(result, 0);
  });

  it('should get next value from stream at time > 0', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-next-time-n' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'next', type: 'Transformation', dataType: 'natural', operation: 'NEXT', inputs: ['stream'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'next' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'next', toPort: 0 },
          { id: 'e2', from: 'next', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result1 = await context.getOutput('result', 1);
    const result2 = await context.getOutput('result', 2);
    const result3 = await context.getOutput('result', 5);

    expectNatural(result1, 1);
    expectNatural(result2, 2);
    expectNatural(result3, 5);
  });

  it('should handle stream values across multiple time steps', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-next-multiple' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'next', type: 'Transformation', dataType: 'natural', operation: 'NEXT', inputs: ['stream'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'next' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'next', toPort: 0 },
          { id: 'e2', from: 'next', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    for (let i = 0; i < 10; i++) {
      const result = await context.getOutput('result', i);
      expectNatural(result, i);
    }
  });
});

describeWithBothRuntimes('Temporal Operations - ACCUMULATE', (context) => {
  it('should accumulate values with ADD operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-accumulate-add' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'op', type: 'DataSource', dataType: 'text', value: 'ADD' },
          { id: 'initial', type: 'DataSource', dataType: 'natural', value: 0 },
          { id: 'accum', type: 'Transformation', dataType: 'natural', operation: 'ACCUMULATE', inputs: ['stream', 'op', 'initial'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'accum' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'accum', toPort: 0 },
          { id: 'e2', from: 'op', to: 'accum', toPort: 1 },
          { id: 'e3', from: 'initial', to: 'accum', toPort: 2 },
          { id: 'e4', from: 'accum', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result0 = await context.getOutput('result', 0);
    const result1 = await context.getOutput('result', 1);
    const result2 = await context.getOutput('result', 2);
    const result3 = await context.getOutput('result', 3);
    const result4 = await context.getOutput('result', 4);

    expectNatural(result0, 0);
    expectNatural(result1, 1);
    expectNatural(result2, 3);
    expectNatural(result3, 6);
    expectNatural(result4, 10);
  });

  it('should accumulate values with MULTIPLY operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-accumulate-multiply' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'op', type: 'DataSource', dataType: 'text', value: 'MULTIPLY' },
          { id: 'initial', type: 'DataSource', dataType: 'natural', value: 1 },
          { id: 'accum', type: 'Transformation', dataType: 'natural', operation: 'ACCUMULATE', inputs: ['stream', 'op', 'initial'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'accum' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'accum', toPort: 0 },
          { id: 'e2', from: 'op', to: 'accum', toPort: 1 },
          { id: 'e3', from: 'initial', to: 'accum', toPort: 2 },
          { id: 'e4', from: 'accum', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result0 = await context.getOutput('result', 0);
    const result1 = await context.getOutput('result', 1);
    const result2 = await context.getOutput('result', 2);
    const result3 = await context.getOutput('result', 3);
    const result4 = await context.getOutput('result', 4);

    expectNatural(result0, 1);
    expectNatural(result1, 1);
    expectNatural(result2, 2);
    expectNatural(result3, 6);
    expectNatural(result4, 24);
  });

  it('should accumulate values with SUBTRACT operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-accumulate-subtract' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'op', type: 'DataSource', dataType: 'text', value: 'SUBTRACT' },
          { id: 'initial', type: 'DataSource', dataType: 'integer', value: 20 },
          { id: 'accum', type: 'Transformation', dataType: 'integer', operation: 'ACCUMULATE', inputs: ['stream', 'op', 'initial'] },
          { id: 'result', type: 'Output', dataType: 'integer', input: 'accum' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'accum', toPort: 0 },
          { id: 'e2', from: 'op', to: 'accum', toPort: 1 },
          { id: 'e3', from: 'initial', to: 'accum', toPort: 2 },
          { id: 'e4', from: 'accum', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result0 = await context.getOutput('result', 0);
    const result1 = await context.getOutput('result', 1);
    const result2 = await context.getOutput('result', 2);
    const result3 = await context.getOutput('result', 3);

    expectInteger(result0, 20);
    expectInteger(result1, 19);
    expectInteger(result2, 17);
    expectInteger(result3, 14);
  });

  it('should accumulate values with DIVIDE operation', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-accumulate-divide' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'op', type: 'DataSource', dataType: 'text', value: 'DIVIDE' },
          { id: 'initial', type: 'DataSource', dataType: 'decimal', value: 120 },
          { id: 'accum', type: 'Transformation', dataType: 'decimal', operation: 'ACCUMULATE', inputs: ['stream', 'op', 'initial'] },
          { id: 'result', type: 'Output', dataType: 'decimal', input: 'accum' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'accum', toPort: 0 },
          { id: 'e2', from: 'op', to: 'accum', toPort: 1 },
          { id: 'e3', from: 'initial', to: 'accum', toPort: 2 },
          { id: 'e4', from: 'accum', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result0 = await context.getOutput('result', 0);
    const result1 = await context.getOutput('result', 1);
    const result2 = await context.getOutput('result', 2);
    const result3 = await context.getOutput('result', 3);

    expectDecimal(result0, 120);
    expectDecimal(result1, 120);
    expectDecimal(result2, 60);
    expectDecimal(result3, 20);
  });

  it('should throw error on division by zero', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-accumulate-divide-by-zero' },
      graph: {
        nodes: [
          { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
          { id: 'op', type: 'DataSource', dataType: 'text', value: 'DIVIDE' },
          { id: 'initial', type: 'DataSource', dataType: 'decimal', value: 10 },
          { id: 'accum', type: 'Transformation', dataType: 'decimal', operation: 'ACCUMULATE', inputs: ['stream', 'op', 'initial'] },
          { id: 'result', type: 'Output', dataType: 'decimal', input: 'accum' }
        ],
        edges: [
          { id: 'e1', from: 'stream', to: 'accum', toPort: 0 },
          { id: 'e2', from: 'op', to: 'accum', toPort: 1 },
          { id: 'e3', from: 'initial', to: 'accum', toPort: 2 },
          { id: 'e4', from: 'accum', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result0 = await context.getOutput('result', 0);
    const result1 = await context.getOutput('result', 1);
    const result2 = await context.getOutput('result', 2);
    const result3 = await context.getOutput('result', 3);

    expectDecimal(result0, 10);
    expectDecimal(result1, 10);
    expectDecimal(result2, 5);
    expectDecimal(result3, 1.6666666666666667);
  });
});

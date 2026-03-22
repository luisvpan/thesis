import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes } from '../../../runtime/src/test-utils';
import { expectNatural } from '../../../runtime/src/test-utils';
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

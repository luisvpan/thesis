import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes } from '../../../runtime/src/test-utils';
import type { DataflowProgram } from '@dataflow/shared/types';

describeWithBothRuntimes('Ordering Operations - SORT', (context) => {
  it('should sort natural numbers in ascending order', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-sort-natural' },
      graph: {
        nodes: [
          { id: 'numbers', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 5 },
            { kind: 'natural', value: 2 },
            { kind: 'natural', value: 8 },
            { kind: 'natural', value: 1 },
            { kind: 'natural', value: 9 },
            { kind: 'natural', value: 3 },
            { kind: 'natural', value: 7 }
          ]}},
          { id: 'sorted', type: 'Transformation', dataType: 'set', operation: 'SORT', inputs: ['numbers'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'sorted' }
        ],
        edges: [
          { id: 'e1', from: 'numbers', to: 'sorted', toPort: 0 },
          { id: 'e2', from: 'sorted', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(7);
    expect(result.elements[0]).toEqual({ kind: 'natural', value: 1 });
    expect(result.elements[6]).toEqual({ kind: 'natural', value: 9 });
  });

  it('should sort integers (including negatives)', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-sort-integer' },
      graph: {
        nodes: [
          { id: 'numbers', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'integer', value: -5 },
            { kind: 'integer', value: 2 },
            { kind: 'integer', value: -3 },
            { kind: 'integer', value: 8 },
            { kind: 'integer', value: -1 },
            { kind: 'integer', value: 0 }
          ]}},
          { id: 'sorted', type: 'Transformation', dataType: 'set', operation: 'SORT', inputs: ['numbers'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'sorted' }
        ],
        edges: [
          { id: 'e1', from: 'numbers', to: 'sorted', toPort: 0 },
          { id: 'e2', from: 'sorted', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(6);
    expect(result.elements[0]).toEqual({ kind: 'integer', value: -5 });
    expect(result.elements[5]).toEqual({ kind: 'integer', value: 8 });
  });

  it('should sort decimals', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-sort-decimal' },
      graph: {
        nodes: [
          { id: 'numbers', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'decimal', value: 1.5 },
            { kind: 'decimal', value: 2.5 },
            { kind: 'decimal', value: 0.5 },
            { kind: 'decimal', value: 3.5 },
            { kind: 'decimal', value: 4.5 }
          ]}},
          { id: 'sorted', type: 'Transformation', dataType: 'set', operation: 'SORT', inputs: ['numbers'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'sorted' }
        ],
        edges: [
          { id: 'e1', from: 'numbers', to: 'sorted', toPort: 0 },
          { id: 'e2', from: 'sorted', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(5);
    expect(result.elements[0]).toEqual({ kind: 'decimal', value: 0.5 });
    expect(result.elements[4]).toEqual({ kind: 'decimal', value: 4.5 });
  });

  it('should sort fractions', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-sort-fraction' },
      graph: {
        nodes: [
          { id: 'fractions', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'fraction', numerator: 1, denominator: 2 },
            { kind: 'fraction', numerator: 3, denominator: 4 },
            { kind: 'fraction', numerator: 1, denominator: 3 },
            { kind: 'fraction', numerator: 5, denominator: 6 }
          ]}},
          { id: 'sorted', type: 'Transformation', dataType: 'set', operation: 'SORT', inputs: ['fractions'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'sorted' }
        ],
        edges: [
          { id: 'e1', from: 'fractions', to: 'sorted', toPort: 0 },
          { id: 'e2', from: 'sorted', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(4);
    expect(result.elements[0]).toEqual({ kind: 'fraction', numerator: 1, denominator: 3 });
    expect(result.elements[3]).toEqual({ kind: 'fraction', numerator: 5, denominator: 6 });
  });
});

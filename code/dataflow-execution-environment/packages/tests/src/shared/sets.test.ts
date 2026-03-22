import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes } from '../../../runtime/src/test-utils';
import type { DataflowProgram } from '@dataflow/shared/types';

describeWithBothRuntimes('Set Operations - INTERSECTION', (context) => {
  it('should find intersection of natural number sets', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-intersection-natural' },
      graph: {
        nodes: [
          { id: 'set1', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 1 },
            { kind: 'natural', value: 2 },
            { kind: 'natural', value: 3 },
            { kind: 'natural', value: 4 }
          ]}},
          { id: 'set2', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 3 },
            { kind: 'natural', value: 4 },
            { kind: 'natural', value: 5 },
            { kind: 'natural', value: 6 }
          ]}},
          { id: 'intersect', type: 'Transformation', dataType: 'set', operation: 'INTERSECTION', inputs: ['set1', 'set2'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'intersect' }
        ],
        edges: [
          { id: 'e1', from: 'set1', to: 'intersect', toPort: 0 },
          { id: 'e2', from: 'set2', to: 'intersect', toPort: 1 },
          { id: 'e3', from: 'intersect', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });

  it('should find intersection with no common elements', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-intersection-empty' },
      graph: {
        nodes: [
          { id: 'set1', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 1 },
            { kind: 'natural', value: 2 }
          ]}},
          { id: 'set2', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 3 },
            { kind: 'natural', value: 4 }
          ]}},
          { id: 'intersect', type: 'Transformation', dataType: 'set', operation: 'INTERSECTION', inputs: ['set1', 'set2'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'intersect' }
        ],
        edges: [
          { id: 'e1', from: 'set1', to: 'intersect', toPort: 0 },
          { id: 'e2', from: 'set2', to: 'intersect', toPort: 1 },
          { id: 'e3', from: 'intersect', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(0);
  });
});

describeWithBothRuntimes('Set Operations - DIFFERENCE', (context) => {
  it('should find difference of natural number sets', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-difference-natural' },
      graph: {
        nodes: [
          { id: 'set1', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 1 },
            { kind: 'natural', value: 2 },
            { kind: 'natural', value: 3 },
            { kind: 'natural', value: 4 }
          ]}},
          { id: 'set2', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 2 },
            { kind: 'natural', value: 4 }
          ]}},
          { id: 'diff', type: 'Transformation', dataType: 'set', operation: 'DIFFERENCE', inputs: ['set1', 'set2'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'diff' }
        ],
        edges: [
          { id: 'e1', from: 'set1', to: 'diff', toPort: 0 },
          { id: 'e2', from: 'set2', to: 'diff', toPort: 1 },
          { id: 'e3', from: 'diff', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });

  it('should find difference with no common elements', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-difference-all' },
      graph: {
        nodes: [
          { id: 'set1', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 1 },
            { kind: 'natural', value: 2 },
            { kind: 'natural', value: 3 }
          ]}},
          { id: 'set2', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 4 },
            { kind: 'natural', value: 5 }
          ]}},
          { id: 'diff', type: 'Transformation', dataType: 'set', operation: 'DIFFERENCE', inputs: ['set1', 'set2'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'diff' }
        ],
        edges: [
          { id: 'e1', from: 'set1', to: 'diff', toPort: 0 },
          { id: 'e2', from: 'set2', to: 'diff', toPort: 1 },
          { id: 'e3', from: 'diff', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(3);
  });
});

describeWithBothRuntimes('Set Operations - UNION', (context) => {
  it('should union natural number sets without duplicates', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-union-natural' },
      graph: {
        nodes: [
          { id: 'set1', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 1 },
            { kind: 'natural', value: 2 },
            { kind: 'natural', value: 3 }
          ]}},
          { id: 'set2', type: 'DataSource', dataType: 'set', value: { kind: 'set', elements: [
            { kind: 'natural', value: 3 },
            { kind: 'natural', value: 4 },
            { kind: 'natural', value: 5 }
          ]}},
          { id: 'union', type: 'Transformation', dataType: 'set', operation: 'UNION', inputs: ['set1', 'set2'] },
          { id: 'result', type: 'Output', dataType: 'set', input: 'union' }
        ],
        edges: [
          { id: 'e1', from: 'set1', to: 'union', toPort: 0 },
          { id: 'e2', from: 'set2', to: 'union', toPort: 1 },
          { id: 'e3', from: 'union', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(5);
  });
});

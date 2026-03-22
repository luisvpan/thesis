import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes } from '../../../runtime/src/test-utils';
import type { DataflowProgram } from '@dataflow/shared/types';

describeWithBothRuntimes('Filtering Operations - FILTER_BY_SIZE', (context) => {
  it('should filter shapes by size', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-filter-size' },
      graph: {
        nodes: [
          { id: 'shapes', type: 'DataSource', dataType: 'set<shape>' as any, value: { kind: 'set', elements: [
            { kind: 'shape', type: 'circle', size: 'small', color: 'red' },
            { kind: 'shape', type: 'square', size: 'large', color: 'blue' },
            { kind: 'shape', type: 'triangle', size: 'small', color: 'green' }
          ]}},
          { id: 'target_size', type: 'DataSource', dataType: 'text', value: 'small' },
          { id: 'filtered', type: 'Transformation', dataType: 'set<shape>' as any, operation: 'FILTER_BY_SIZE', inputs: ['shapes', 'target_size'] },
          { id: 'result', type: 'Output', dataType: 'set<shape>' as any, input: 'filtered' }
        ],
        edges: [
          { id: 'e1', from: 'shapes', to: 'filtered', toPort: 0 },
          { id: 'e2', from: 'target_size', to: 'filtered', toPort: 1 },
          { id: 'e3', from: 'filtered', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });
});

describeWithBothRuntimes('Filtering Operations - FILTER_BY_COLOR', (context) => {
  it('should filter shapes by color', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-filter-color' },
      graph: {
        nodes: [
          { id: 'shapes', type: 'DataSource', dataType: 'set<shape>' as any, value: { kind: 'set', elements: [
            { kind: 'shape', type: 'circle', size: 'small', color: 'red' },
            { kind: 'shape', type: 'square', size: 'large', color: 'blue' },
            { kind: 'shape', type: 'triangle', size: 'small', color: 'red' }
          ]}},
          { id: 'target_color', type: 'DataSource', dataType: 'text', value: 'red' },
          { id: 'filtered', type: 'Transformation', dataType: 'set<shape>' as any, operation: 'FILTER_BY_COLOR', inputs: ['shapes', 'target_color'] },
          { id: 'result', type: 'Output', dataType: 'set<shape>' as any, input: 'filtered' }
        ],
        edges: [
          { id: 'e1', from: 'shapes', to: 'filtered', toPort: 0 },
          { id: 'e2', from: 'target_color', to: 'filtered', toPort: 1 },
          { id: 'e3', from: 'filtered', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });
});

describeWithBothRuntimes('Filtering Operations - FILTER_BY_TYPE', (context) => {
  it('should filter shapes by type', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-filter-type' },
      graph: {
        nodes: [
          { id: 'shapes', type: 'DataSource', dataType: 'set<shape>' as any, value: { kind: 'set', elements: [
            { kind: 'shape', type: 'circle', size: 'small', color: 'red' },
            { kind: 'shape', type: 'circle', size: 'large', color: 'blue' },
            { kind: 'shape', type: 'square', size: 'small', color: 'green' }
          ]}},
          { id: 'target_type', type: 'DataSource', dataType: 'text', value: 'circle' },
          { id: 'filtered', type: 'Transformation', dataType: 'set<shape>' as any, operation: 'FILTER_BY_TYPE', inputs: ['shapes', 'target_type'] },
          { id: 'result', type: 'Output', dataType: 'set<shape>' as any, input: 'filtered' }
        ],
        edges: [
          { id: 'e1', from: 'shapes', to: 'filtered', toPort: 0 },
          { id: 'e2', from: 'target_type', to: 'filtered', toPort: 1 },
          { id: 'e3', from: 'filtered', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });
});

describeWithBothRuntimes('Filtering Operations - FILTER_BY_TASTE', (context) => {
  it('should filter foods by taste', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-filter-taste' },
      graph: {
        nodes: [
          { id: 'foods', type: 'DataSource', dataType: 'set<food>' as any, value: { kind: 'set', elements: [
            { kind: 'food', taste: 'sweet', color: 'red' },
            { kind: 'food', taste: 'salty', color: 'blue' },
            { kind: 'food', taste: 'sweet', color: 'green' }
          ]}},
          { id: 'target_taste', type: 'DataSource', dataType: 'text', value: 'sweet' },
          { id: 'filtered', type: 'Transformation', dataType: 'set<food>' as any, operation: 'FILTER_BY_TASTE', inputs: ['foods', 'target_taste'] },
          { id: 'result', type: 'Output', dataType: 'set<food>' as any, input: 'filtered' }
        ],
        edges: [
          { id: 'e1', from: 'foods', to: 'filtered', toPort: 0 },
          { id: 'e2', from: 'target_taste', to: 'filtered', toPort: 1 },
          { id: 'e3', from: 'filtered', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });
});

describeWithBothRuntimes('Filtering Operations - FILTER_BY_AGE_GROUP', (context) => {
  it('should filter people by age group', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-filter-age-group' },
      graph: {
        nodes: [
          { id: 'people', type: 'DataSource', dataType: 'set<person>' as any, value: { kind: 'set', elements: [
            { kind: 'person', ageGroup: 'child', gender: 'male' },
            { kind: 'person', ageGroup: 'adult', gender: 'female' },
            { kind: 'person', ageGroup: 'child', gender: 'female' }
          ]}},
          { id: 'target_age', type: 'DataSource', dataType: 'text', value: 'child' },
          { id: 'filtered', type: 'Transformation', dataType: 'set<person>' as any, operation: 'FILTER_BY_AGE_GROUP', inputs: ['people', 'target_age'] },
          { id: 'result', type: 'Output', dataType: 'set<person>' as any, input: 'filtered' }
        ],
        edges: [
          { id: 'e1', from: 'people', to: 'filtered', toPort: 0 },
          { id: 'e2', from: 'target_age', to: 'filtered', toPort: 1 },
          { id: 'e3', from: 'filtered', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });
});

describeWithBothRuntimes('Filtering Operations - FILTER_BY_GENDER', (context) => {
  it('should filter people by gender', async () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-filter-gender' },
      graph: {
        nodes: [
          { id: 'people', type: 'DataSource', dataType: 'set<person>' as any, value: { kind: 'set', elements: [
            { kind: 'person', ageGroup: 'child', gender: 'male' },
            { kind: 'person', ageGroup: 'adult', gender: 'female' },
            { kind: 'person', ageGroup: 'child', gender: 'male' }
          ]}},
          { id: 'target_gender', type: 'DataSource', dataType: 'text', value: 'male' },
          { id: 'filtered', type: 'Transformation', dataType: 'set<person>' as any, operation: 'FILTER_BY_GENDER', inputs: ['people', 'target_gender'] },
          { id: 'result', type: 'Output', dataType: 'set<person>' as any, input: 'filtered' }
        ],
        edges: [
          { id: 'e1', from: 'people', to: 'filtered', toPort: 0 },
          { id: 'e2', from: 'target_gender', to: 'filtered', toPort: 1 },
          { id: 'e3', from: 'filtered', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = await context.getOutput('result');
    expect(result.kind).toBe('set');
    expect(result.elements).toHaveLength(2);
  });
});

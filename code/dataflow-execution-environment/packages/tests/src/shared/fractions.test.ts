import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes } from '../../../runtime/src/test-utils';
import { expectFraction, expectBoolean } from '../../../runtime/src/test-utils';
import type { DataflowProgram } from '@dataflow/shared/types';

describeWithBothRuntimes('Fraction Operations - ADD', (context) => {
  it('should execute ADD operation with fractions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-add' },
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

    context.loadProgram(program);
    const result = context.getOutput('result');
    expectFraction(result, { numerator: 3, denominator: 4 });
  });

  it('should ADD fractions that simplify to whole number', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-add-whole' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 2, denominator: 3 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 3 } },
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

    context.loadProgram(program);
    const result = context.getOutput('result');
    expectFraction(result, { numerator: 1, denominator: 1 });
  });
});

describeWithBothRuntimes('Fraction Operations - SUBTRACT', (context) => {
  it('should execute SUBTRACT operation with fractions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-subtract' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 3, denominator: 4 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 4 } },
          { id: 'sub', type: 'Transformation', dataType: 'fraction', operation: 'SUBTRACT', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'fraction', input: 'sub' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'sub', toPort: 0 },
          { id: 'e2', from: 'b', to: 'sub', toPort: 1 },
          { id: 'e3', from: 'sub', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = context.getOutput('result');
    expectFraction(result, { numerator: 1, denominator: 2 });
  });
});

describeWithBothRuntimes('Fraction Operations - MULTIPLY', (context) => {
  it('should execute MULTIPLY operation with fractions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-multiply' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 2, denominator: 3 } },
          { id: 'mul', type: 'Transformation', dataType: 'fraction', operation: 'MULTIPLY', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'fraction', input: 'mul' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'mul', toPort: 0 },
          { id: 'e2', from: 'b', to: 'mul', toPort: 1 },
          { id: 'e3', from: 'mul', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = context.getOutput('result');
    expectFraction(result, { numerator: 1, denominator: 3 });
  });
});

describeWithBothRuntimes('Fraction Operations - DIVIDE', (context) => {
  it('should execute DIVIDE operation with fractions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-divide' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 4 } },
          { id: 'div', type: 'Transformation', dataType: 'fraction', operation: 'DIVIDE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'fraction', input: 'div' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'div', toPort: 0 },
          { id: 'e2', from: 'b', to: 'div', toPort: 1 },
          { id: 'e3', from: 'div', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = context.getOutput('result');
    expectFraction(result, { numerator: 2, denominator: 1 });
  });

  it('should error on division by zero in DIVIDE', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-divide-by-zero' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 0, denominator: 1 } },
          { id: 'div', type: 'Transformation', dataType: 'fraction', operation: 'DIVIDE', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'fraction', input: 'div' }
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
      expect(() => context.execute()).toThrow('DIVIDE: Division by zero');
    } else {
      const result = context.execute() as { nodeStates: Map<string, any>; changedNodes: string[] };
      const errorState = result.nodeStates.get('result');
      expect(errorState?.status).toBe('error');
      expect(errorState?.error).toContain('Division by zero');
    }
  });
});

describeWithBothRuntimes('Fraction Operations - COMPARE', (context) => {
  it('should execute COMPARE operation with equal fractions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-compare-equal' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 2, denominator: 4 } },
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
    const result = context.getOutput('result');
    expectBoolean(result, true);
  });

  it('should execute COMPARE operation with different fractions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-compare-different' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 3 } },
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
    const result = context.getOutput('result');
    expectBoolean(result, false);
  });
});

describeWithBothRuntimes('Fraction Operations - Edge Cases', (context) => {
  it('should simplify fractions automatically', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-simplify' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 2, denominator: 4 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 3, denominator: 6 } },
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

    context.loadProgram(program);
    const result = context.getOutput('result');
    expectFraction(result, { numerator: 1, denominator: 1 });
  });

  it('should handle negative fractions', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-negative' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 3, denominator: 4 } },
          { id: 'sub', type: 'Transformation', dataType: 'fraction', operation: 'SUBTRACT', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'fraction', input: 'sub' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'sub', toPort: 0 },
          { id: 'e2', from: 'b', to: 'sub', toPort: 1 },
          { id: 'e3', from: 'sub', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = context.getOutput('result');
    expectFraction(result, { numerator: -1, denominator: 4 });
  });

  it('should error on zero denominator in fraction inputs', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-fraction-zero-denominator' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 2 } },
          { id: 'b', type: 'DataSource', dataType: 'fraction', value: { kind: 'fraction', numerator: 1, denominator: 0 } },
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

    context.loadProgram(program);
    
    if (context.runtime.constructor.name === 'Runtime') {
      expect(() => context.execute()).toThrow('ADD: Denominator cannot be zero');
    } else {
      const result = context.execute() as { nodeStates: Map<string, any>; changedNodes: string[] };
      const errorState = result.nodeStates.get('result');
      expect(errorState?.status).toBe('error');
      expect(errorState?.error).toContain('Denominator cannot be zero');
    }
  });
});

import { describe, it, expect } from 'bun:test';

import { Compiler } from './compiler';
import type { DataflowProgram } from '@dataflow/shared/types';

describe('Compiler - Lexing and Parsing', () => {
  it('should parse simple source statement', () => {
    const compiler = new Compiler();
    const source = 'source a: natural = 5;';
    const ast = compiler.parse(source);
    
    expect(ast).toBeDefined();
    expect(ast.type).toBe('Program');
    expect(ast.statements).toHaveLength(1);
    expect(ast.statements[0]).toMatchObject({
      type: 'SourceStatement',
      id: 'a',
      dataType: 'natural'
    });
  });

  it('should parse multiple source statements', () => {
    const compiler = new Compiler();
    const source = 'source a: natural = 5;\nsource b: natural = 3;';
    const ast = compiler.parse(source);
    
    expect(ast.statements).toHaveLength(2);
    expect(ast.statements[0].id).toBe('a');
    expect(ast.statements[1].id).toBe('b');
  });

  it('should parse transform statement with ADD', () => {
    const compiler = new Compiler();
    const source = 'transform sum: natural = ADD(a, b);';
    const ast = compiler.parse(source);
    
    expect(ast.statements).toHaveLength(1);
    expect(ast.statements[0]).toMatchObject({
      type: 'TransformStatement',
      id: 'sum',
      dataType: 'natural',
      operation: 'ADD'
    });
    expect(ast.statements[0].inputs).toHaveLength(2);
  });

  it('should parse output statement', () => {
    const compiler = new Compiler();
    const source = 'output result: natural = sum;';
    const ast = compiler.parse(source);
    
    expect(ast.statements).toHaveLength(1);
    expect(ast.statements[0]).toMatchObject({
      type: 'OutputStatement',
      id: 'result',
      dataType: 'natural',
      input: 'sum'
    });
  });

  it('should parse complete program with all statement types', () => {
    const compiler = new Compiler();
    const source = `
      source a: natural = 3;
      source b: natural = 2;
      transform sum: natural = ADD(a, b);
      output result: natural = sum;
    `;
    const ast = compiler.parse(source);
    
    expect(ast.statements).toHaveLength(4);
    expect(ast.statements[0].type).toBe('SourceStatement');
    expect(ast.statements[1].type).toBe('SourceStatement');
    expect(ast.statements[2].type).toBe('TransformStatement');
    expect(ast.statements[3].type).toBe('OutputStatement');
  });

  it('should handle comments in source code', () => {
    const compiler = new Compiler();
    const source = `
      
      source a: natural = 3;
      
      output result: natural = a;
    `;
    const ast = compiler.parse(source);
    
    expect(ast.statements).toHaveLength(2);
  });
});

describe('Compiler - Validation', () => {
  it('should validate correct program', () => {
    const compiler = new Compiler();
    
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
    
    const result = compiler.compile(program);
    
    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
    expect(result.graph).toBeDefined();
  });

  it('should detect duplicate identifiers', () => {
    const compiler = new Compiler();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 2 }
        ],
        edges: []
      }
    };
    
    const result = compiler.compile(program);
    
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('DUPLICATE_IDENTIFIER');
    expect(result.errors[0].childMessage).toBeDefined();
  });

  it('should detect undefined references', () => {
    const compiler = new Compiler();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'add', toPort: 0 }
        ]
      }
    };
    
    const result = compiler.compile(program);
    
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('UNDEFINED_IDENTIFIER');
  });

  it('should detect cycles', () => {
    const compiler = new Compiler();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['c', 'b'] },
          { id: 'b', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'c'] },
          { id: 'c', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'b', toPort: 0 },
          { id: 'e2', from: 'b', to: 'c', toPort: 0 },
          { id: 'e3', from: 'c', to: 'a', toPort: 0 }
        ]
      }
    };
    
    const result = compiler.compile(program);
    
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('CYCLE_DETECTED');
    expect(result.errors[0].childMessage).toBeDefined();
  });

  it('should detect wrong arity', () => {
    const compiler = new Compiler();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a'] }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'add', toPort: 0 }
        ]
      }
    };
    
    const result = compiler.compile(program);
    
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('WRONG_ARITY');
    expect(result.errors[0].childMessage).toBeDefined();
  });

  it('should detect unknown operation', () => {
    const compiler = new Compiler();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'badop', type: 'Transformation', dataType: 'natural', operation: 'UNKNOWN_OP', inputs: ['a', 'a'] }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'badop', toPort: 0 },
          { id: 'e2', from: 'a', to: 'badop', toPort: 1 }
        ]
      }
    };
    
    const result = compiler.compile(program);
    
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('UNKNOWN_OPERATION');
    expect(result.errors[0].childMessage).toBeDefined();
  });
});

import { describe, it, expect } from 'bun:test';

import { Compiler } from './compiler';
import type { OutputStatement, SourceStatement, TransformStatement } from './ast';

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
      dataType: 'natural',
      value: 5
    } satisfies SourceStatement);
  });

  it('should parse multiple source statements', () => {
    const compiler = new Compiler();
    const source = 'source a: natural = 5;\nsource b: natural = 3;';
    const ast = compiler.parse(source);

    expect(ast.statements).toHaveLength(2);
    expect(ast.statements[0].id).toBe('a');
    expect((ast.statements[0] as SourceStatement).value).toBe(5);
    expect(ast.statements[1].id).toBe('b');
    expect((ast.statements[1] as SourceStatement).value).toBe(3);
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
      operation: 'ADD',
      inputs: ['a', 'b']
    } satisfies TransformStatement);
    expect((ast.statements[0] as TransformStatement).inputs).toHaveLength(2);
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
    } satisfies OutputStatement);
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
      /* This is a comment */
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
    const source = `
      source a: natural = 3;
      source b: natural = 2;
      transform add: natural = ADD(a, b);
      output result: natural = add;
    `;

    const result = compiler.compile(source);

    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
    expect(result.program).toBeDefined();
  });

  it('should detect duplicate identifiers', () => {
    const compiler = new Compiler();
    const source = `
      source a: natural = 3;
      source a: natural = 2;
    `;

    const result = compiler.compile(source);

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('DUPLICATE_IDENTIFIER');
    expect(result.errors[0].childMessage).toBeDefined();
  });

  it('should detect undefined references', () => {
    const compiler = new Compiler();
    const source = `
      source a: natural = 3;
      transform add: natural = ADD(a, b);
    `;

    const result = compiler.compile(source);

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('UNDEFINED_IDENTIFIER');
  });

  it('should detect cycles', () => {
    const compiler = new Compiler();
    const source = `
      source x: natural = 3;
      source y: natural = 2;
      source z: natural = 1;
      transform a: natural = ADD(z, b);
      transform b: natural = ADD(a, c);
      transform c: natural = ADD(a, b);
    `;

    const result = compiler.compile(source);

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('CYCLE_DETECTED');
    expect(result.errors[0].childMessage).toBeDefined();
  });

  it('should detect wrong arity', () => {
    const compiler = new Compiler();
    const source = `
      source a: natural = 3;
      transform add: natural = ADD(a);
    `;

    const result = compiler.compile(source);

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('WRONG_ARITY');
    expect(result.errors[0].childMessage).toBeDefined();
  });

  it('should detect unknown operation', () => {
    const compiler = new Compiler();
    const source = `
      source a: natural = 3;
      transform badop: natural = UNKNOWN_OP(a, a);
    `;

    const result = compiler.compile(source);

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].code).toBe('UNKNOWN_OPERATION');
    expect(result.errors[0].childMessage).toBeDefined();
  });
});

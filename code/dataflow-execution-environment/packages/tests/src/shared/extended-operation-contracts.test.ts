import { describe, it, expect } from 'bun:test';
import { Compiler } from '@dataflow/compiler';
import { Runtime } from '@dataflow/runtime';

describe('Extended Operation Contracts', () => {
  describe('COMPARE with Text and Boolean', () => {
    it('should compare Text values - equal', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source text1: text = "hello";
        source text2: text = "hello";

        transform cmp: boolean = COMPARE(text1, text2);

        output result: boolean = cmp;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: 'boolean', value: true });
    });

    it('should compare Text values - not equal', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source text1: text = "hello";
        source text2: text = "world";

        transform cmp: boolean = COMPARE(text1, text2);

        output result: boolean = cmp;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: 'boolean', value: false });
    });

    it('should compare Boolean values - equal true', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source bool1: boolean = true;
        source bool2: boolean = true;

        transform cmp: boolean = COMPARE(bool1, bool2);

        output result: boolean = cmp;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: 'boolean', value: true });
    });

    it('should compare Boolean values - equal false', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source bool1: boolean = false;
        source bool2: boolean = false;

        transform cmp: boolean = COMPARE(bool1, bool2);

        output result: boolean = cmp;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: 'boolean', value: true });
    });

    it('should compare Boolean values - not equal', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source bool1: boolean = true;
        source bool2: boolean = false;

        transform cmp: boolean = COMPARE(bool1, bool2);

        output result: boolean = cmp;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: 'boolean', value: false });
    });
  });

  describe('FILTER with Integer, Decimal, Fraction', () => {
    it('should filter Integer set', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<integer> = {-5, 2, -3, 8, -1, 0};

        transform filtered: set<integer> = FILTER(numbers, -3);

        output result: set<integer> = filtered;
      `;

      const compileResult = compiler.compile(source);
      if (!compileResult.success) {
        console.log('Compilation errors:', JSON.stringify(compileResult.errors, null, 2));
      }
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe('set');
      expect(result.elements).toHaveLength(1);
      expect(result.elements[0]).toEqual({ kind: 'integer', value: -3 });
    });

    it('should filter Decimal set', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<decimal> = {1.5, 2.5, 3.5, 4.5};

        transform filtered: set<decimal> = FILTER(numbers, 3.5);

        output result: set<decimal> = filtered;
      `;

      const compileResult = compiler.compile(source);
      if (!compileResult.success) {
        console.log('Decimal compilation errors:', JSON.stringify(compileResult.errors, null, 2));
      }
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe('set');
      expect(result.elements).toHaveLength(1);
      expect(result.elements[0]).toEqual({ kind: 'decimal', value: 3.5 });
    });

    it('should filter Fraction set', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source fractions: set<fraction> = {1/2, 3/4, 1/3, 5/6};

        transform filtered: set<fraction> = FILTER(fractions, 3/4);

        output result: set<fraction> = filtered;
      `;

      const compileResult = compiler.compile(source);
      if (!compileResult.success) {
        console.log('Fraction compilation errors:', JSON.stringify(compileResult.errors, null, 2));
      }
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe('set');
      expect(result.elements).toHaveLength(1);
      expect(result.elements[0]).toEqual({ kind: 'fraction', numerator: 3, denominator: 4 });
    });
  });

  describe('SORT with Integer, Decimal, Fraction', () => {
    it('should sort Integer set', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<integer> = {-5, 2, -3, 8, -1, 0};

        transform sorted: set<integer> = SORT(numbers);

        output result: set<integer> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe('set');
      expect(result.elements).toEqual([
        { kind: 'integer', value: -5 },
        { kind: 'integer', value: -3 },
        { kind: 'integer', value: -1 },
        { kind: 'integer', value: 0 },
        { kind: 'integer', value: 2 },
        { kind: 'integer', value: 8 }
      ]);
    });

    it('should sort Decimal set', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<decimal> = {1.5, 2.5, 0.5, 3.5, 4.5};

        transform sorted: set<decimal> = SORT(numbers);

        output result: set<decimal> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe('set');
      expect(result.elements).toEqual([
        { kind: 'decimal', value: 0.5 },
        { kind: 'decimal', value: 1.5 },
        { kind: 'decimal', value: 2.5 },
        { kind: 'decimal', value: 3.5 },
        { kind: 'decimal', value: 4.5 }
      ]);
    });

    it('should sort Fraction set', async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source fractions: set<fraction> = {1/2, 3/4, 1/3, 5/6};

        transform sorted: set<fraction> = SORT(fractions);

        output result: set<fraction> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe('set');
      expect(result.elements).toEqual([
        { kind: 'fraction', numerator: 1, denominator: 3 },
        { kind: 'fraction', numerator: 1, denominator: 2 },
        { kind: 'fraction', numerator: 3, denominator: 4 },
        { kind: 'fraction', numerator: 5, denominator: 6 }
      ]);
    });
  });
});

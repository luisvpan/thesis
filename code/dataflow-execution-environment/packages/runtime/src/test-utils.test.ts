import { describe, it, expect } from 'bun:test';
import { createRuntimeContext, describeWithBothRuntimes, simpleArithmeticProgram, expectNatural, expectFraction } from './test-utils';

describe('Test Utilities - Runtime Factory', () => {
  it('should create batch runtime context', () => {
    const context = createRuntimeContext('batch');

    expect(context.runtime).toBeDefined();
    expect(context.loadProgram).toBeInstanceOf(Function);
    expect(context.execute).toBeInstanceOf(Function);
    expect(context.getOutput).toBeInstanceOf(Function);
  });

  it('should create incremental runtime context', () => {
    const context = createRuntimeContext('incremental');

    expect(context.runtime).toBeDefined();
    expect(context.loadProgram).toBeInstanceOf(Function);
    expect(context.execute).toBeInstanceOf(Function);
    expect(context.getOutput).toBeInstanceOf(Function);
  });

  it('should execute batch runtime context', async () => {
    const context = createRuntimeContext('batch');
    context.loadProgram(simpleArithmeticProgram);
 
    const result = await context.execute();
    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(1);
    expectNatural(result[0], 5);
  });

  it('should execute incremental runtime context', async () => {
    const context = createRuntimeContext('incremental');
    context.loadProgram(simpleArithmeticProgram);
 
    const result = await context.execute();
    expect(result).toHaveProperty('nodeStates');
    expect(result).toHaveProperty('changedNodes');
    expect(result.nodeStates instanceof Map).toBe(true);
    expect(Array.isArray(result.changedNodes)).toBe(true);
  });

  it('should get output from batch runtime context', async () => {
    const context = createRuntimeContext('batch');
    context.loadProgram(simpleArithmeticProgram);
 
    const result = await context.getOutput('result');
    expectNatural(result, 5);
  });

  it('should get output from incremental runtime context', async () => {
    const context = createRuntimeContext('incremental');
    context.loadProgram(simpleArithmeticProgram);
 
    const result = await context.getOutput('result');
    expectNatural(result, 5);
  });
});

describeWithBothRuntimes('Test Utilities - describeWithBothRuntimes', async (context) => {
  it('should run tests on both runtimes', async () => {
    context.loadProgram(simpleArithmeticProgram);
    const result = await context.getOutput('result');
    expectNatural(result, 5);
  });
});

describe('Test Utilities - Test Fixtures', () => {
  it('simpleArithmeticProgram should be valid DataflowProgram', () => {
    expect(simpleArithmeticProgram).toHaveProperty('metadata');
    expect(simpleArithmeticProgram).toHaveProperty('graph');
    expect(simpleArithmeticProgram.metadata).toHaveProperty('programId');
    expect(simpleArithmeticProgram.graph).toHaveProperty('nodes');
    expect(simpleArithmeticProgram.graph).toHaveProperty('edges');
    expect(Array.isArray(simpleArithmeticProgram.graph.nodes)).toBe(true);
    expect(Array.isArray(simpleArithmeticProgram.graph.edges)).toBe(true);
  });
});

describe('Test Utilities - Test Helpers', () => {
  it('expectNatural should assert correctly', () => {
    const value = { kind: 'natural', value: 42 };
    expect(() => expectNatural(value, 42)).not.toThrow();
  });

  it('expectNatural should throw on mismatch', () => {
    const value = { kind: 'natural', value: 42 };
    expect(() => expectNatural(value, 43)).toThrow();
  });

  it('expectFraction should assert correctly', () => {
    const value = { kind: 'fraction', numerator: 1, denominator: 2 };
    expect(() => expectFraction(value, { numerator: 1, denominator: 2 })).not.toThrow();
  });

  it('expectFraction should throw on mismatch', () => {
    const value = { kind: 'fraction', numerator: 1, denominator: 2 };
    expect(() => expectFraction(value, { numerator: 1, denominator: 3 })).toThrow();
  });
});

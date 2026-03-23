import { expect } from 'bun:test';

export function expectNatural(value: any, expected: number) {
  expect(value).toEqual({ kind: 'natural', value: expected });
}

export function expectInteger(value: any, expected: number) {
  expect(value).toEqual({ kind: 'integer', value: expected });
}

export function expectDecimal(value: any, expected: number) {
  expect(value).toEqual({ kind: 'decimal', value: expected });
}

export function expectFraction(value: any, expected: { numerator: number; denominator: number }) {
  expect(value).toEqual({ kind: 'fraction', ...expected });
}

export function expectBoolean(value: any, expected: boolean) {
  expect(value).toEqual({ kind: 'boolean', value: expected });
}

export function expectText(value: any, expected: string) {
  expect(value).toEqual({ kind: 'text', value: expected });
}

export function expectShape(value: any, expected: { type: string; size: string; color: string }) {
  expect(value).toEqual({ kind: 'shape', ...expected });
}

export function expectCar(value: any, expected: { color: string }) {
  expect(value).toEqual({ kind: 'car', ...expected });
}

export function expectFood(value: any, expected: { taste: string; color: string }) {
  expect(value).toEqual({ kind: 'food', ...expected });
}

export function expectAnimal(value: any, expected: { type: string; color: string }) {
  expect(value).toEqual({ kind: 'animal', ...expected });
}

export function expectPerson(value: any, expected: { ageGroup: string; gender: string }) {
  expect(value).toEqual({ kind: 'person', ...expected });
}

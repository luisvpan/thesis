import { describe, expect, test } from 'bun:test';
import { numberToSpanishWords, replaceDigitsWithSpanishWords } from './spanishNumberWords';

describe('numberToSpanishWords', () => {
  test('0–15', () => {
    expect(numberToSpanishWords(0)).toBe('cero');
    expect(numberToSpanishWords(16)).toBe('dieciseis');
    expect(numberToSpanishWords(15)).toBe('quince');
  });

  test('21 y compuestos', () => {
    expect(numberToSpanishWords(21)).toBe('veintiuno');
    expect(numberToSpanishWords(35)).toBe('treinta y cinco');
  });

  test('centenas', () => {
    expect(numberToSpanishWords(100)).toBe('cien');
    expect(numberToSpanishWords(101)).toBe('ciento uno');
    expect(numberToSpanishWords(540)).toBe('quinientos cuarenta');
    expect(numberToSpanishWords(999)).toBe('novecientos noventa y nueve');
  });

  test('miles y millones', () => {
    expect(numberToSpanishWords(1000)).toBe('mil');
    expect(numberToSpanishWords(1540)).toBe('mil quinientos cuarenta');
    expect(numberToSpanishWords(21000)).toBe('veintiun mil');
    expect(numberToSpanishWords(31000)).toBe('treinta y un mil');
    expect(numberToSpanishWords(1_000_000)).toBe('un millón');
    expect(numberToSpanishWords(2_500_000)).toBe('dos millones quinientos mil');
  });

  test('negativos y decimales', () => {
    expect(numberToSpanishWords(-5)).toBe('menos cinco');
    expect(numberToSpanishWords(2.5)).toBe('dos coma cinco');
    expect(numberToSpanishWords(3.25)).toBe('tres coma veinticinco');
    expect(numberToSpanishWords(1.05)).toBe('uno coma cero cinco');
  });
});

describe('replaceDigitsWithSpanishWords', () => {
  test('reemplaza números en frases', () => {
    expect(replaceDigitsWithSpanishWords('16 objetos')).toBe('dieciseis objetos');
    expect(replaceDigitsWithSpanishWords('540 manzanas')).toBe('quinientos cuarenta manzanas');
  });
});

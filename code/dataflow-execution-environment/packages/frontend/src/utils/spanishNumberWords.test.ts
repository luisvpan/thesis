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
});

describe('replaceDigitsWithSpanishWords', () => {
  test('reemplaza números en frases', () => {
    expect(replaceDigitsWithSpanishWords('16 objetos')).toBe('dieciseis objetos');
  });
});

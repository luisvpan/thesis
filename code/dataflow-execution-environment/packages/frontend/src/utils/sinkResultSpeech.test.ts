import { describe, expect, test } from 'bun:test';
import type { ResultValue } from '@/services/executeProgram';
import { buildSinkResultSpeechText } from './sinkResultSpeech';
import type { ProgramOutputFlowNodeData } from '@/components/dataflow/ProgramOutputFlowNode';

const base: ProgramOutputFlowNodeData = {};

function spoken(resultValue: ResultValue): string | null {
  return buildSinkResultSpeechText({ resultValue }, null, 'abstracto');
}

/** Una bolsa semántica con la descripción que se quiera. */
function semantic(description: string): ResultValue {
  return {
    kind: 'semantic',
    result: { categories: [], totalAmount: 0, description, visualStrip: [] },
  };
}

describe('buildSinkResultSpeechText', () => {
  test('no lee el texto del error de ejecución', () => {
    expect(buildSinkResultSpeechText(base, 'falló', 'abstracto')).toBeNull();
  });

  test('convierte dígitos en descripción semántica', () => {
    expect(spoken(semantic('16 objetos'))).toBe('dieciseis objetos');
  });

  test('número abstracto se pronuncia en palabras', () => {
    expect(spoken({ kind: 'number', value: 16 })).toBe('dieciseis');
  });

  test('un arreglo ordenado se dice en orden', () => {
    expect(
      spoken({
        kind: 'numberArray',
        values: [
          { value: 5, numerator: '5', denominator: '1' },
          { value: 3, numerator: '3', denominator: '1' },
        ],
      })
    ).toBe('cinco, tres');
  });

  test('un booleano se dice', () => {
    expect(spoken({ kind: 'boolean', value: true })).toBe('verdadero');
  });

  test('sin resultado devuelve null', () => {
    expect(buildSinkResultSpeechText(base, null, 'abstracto')).toBeNull();
  });
});

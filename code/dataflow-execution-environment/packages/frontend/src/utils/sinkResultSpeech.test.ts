import { describe, expect, test } from 'bun:test';
import { buildSinkResultSpeechText } from './sinkResultSpeech';
import type { ProgramOutputFlowNodeData } from '@/components/dataflow/ProgramOutputFlowNode';

const base: ProgramOutputFlowNodeData = {};

describe('buildSinkResultSpeechText', () => {
  test('no lee el texto del error de ejecución', () => {
    expect(buildSinkResultSpeechText(base, 'falló', 'abstracto')).toBeNull();
  });

  test('convierte dígitos en descripción semántica', () => {
    const data: ProgramOutputFlowNodeData = { description: '16 objetos' };
    expect(buildSinkResultSpeechText(data, null, 'abstracto')).toBe('dieciseis objetos');
  });

  test('número abstracto se pronuncia en palabras', () => {
    const data: ProgramOutputFlowNodeData = { value: 16 };
    expect(buildSinkResultSpeechText(data, null, 'abstracto')).toBe('dieciseis');
  });

  test('sin resultado devuelve null', () => {
    expect(buildSinkResultSpeechText(base, null, 'abstracto')).toBeNull();
  });
});

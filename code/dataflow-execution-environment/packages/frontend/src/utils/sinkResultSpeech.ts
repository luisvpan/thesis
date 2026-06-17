import type { ProgramOutputFlowNodeData } from '@/components/dataflow/ProgramOutputFlowNode';
import type { ResultViewMode } from '@/components/dataflow/dataflowResultCpa';
import type { SingleCpaObjectMeta } from '@/services/executeProgram';
import { numberToSpanishWords, replaceDigitsWithSpanishWords } from './spanishNumberWords';

function spokenNumber(value: number): string {
  if (Number.isInteger(value) && value >= 0 && value <= 99) {
    return numberToSpanishWords(value);
  }
  return String(value);
}

function singleCpaSpeechText(meta: SingleCpaObjectMeta): string {
  const colorPart = meta.color ? meta.color : '';
  const typeLabels: Record<string, string> = {
    montessori: 'cubos',
    cap: 'tapas',
    stick: 'paletas',
    forma: meta.subtype,
    comida: meta.subtype,
  };
  const label = typeLabels[meta.type] ?? meta.type;
  const qty = numberToSpanishWords(meta.quantity);
  if (colorPart) {
    return `${qty} ${label} ${colorPart}`.trim();
  }
  return `${qty} ${label}`.trim();
}

/** Texto listo para TTS según el resultado del sink; `null` si aún no hay nada que decir. */
export function buildSinkResultSpeechText(
  data: ProgramOutputFlowNodeData,
  executionError: string | null | undefined,
  _viewMode: ResultViewMode
): string | null {
  if (executionError?.trim()) {
    return null;
  }

  if (data.booleanValue !== undefined) {
    return data.booleanValue ? 'verdadero' : 'falso';
  }

  // Ordered array of abstract numbers
  if (data.numberArrayValues && data.numberArrayValues.length > 0) {
    const spoken = data.numberArrayValues.map((item) => spokenNumber(item.value));
    return spoken.join(', ');
  }

  if (data.isSingleCpaObject && data.singleCpaObjectMeta) {
    return singleCpaSpeechText(data.singleCpaObjectMeta);
  }

  if (data.description?.trim()) {
    return replaceDigitsWithSpanishWords(data.description.trim());
  }

  if (data.value !== undefined && Number.isFinite(data.value)) {
    return spokenNumber(data.value);
  }

  return null;
}

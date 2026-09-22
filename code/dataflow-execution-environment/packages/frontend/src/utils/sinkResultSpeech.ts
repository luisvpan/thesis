import type { ProgramOutputFlowNodeData } from '@/components/dataflow/ProgramOutputFlowNode';
import type { ResultViewMode } from '@/components/dataflow/dataflowResultCpa';
import type { SingleCpaObjectMeta } from '@/services/executeProgram';
import { numberToSpanishWords, replaceDigitsWithSpanishWords } from './spanishNumberWords';
import { describeCountedNoun } from './spanishGrammar';

function spokenNumber(value: number): string {
  if (Number.isInteger(value) && value >= 0 && value <= 99) {
    return numberToSpanishWords(value);
  }
  return String(value);
}

// Para montessori/cap/stick el sustantivo real es el tipo (cubo/tapa/paleta),
// no el subtipo que reporta el intérprete (que en esos casos es el color).
const NOUN_KEY_BY_TYPE: Record<string, string> = {
  montessori: 'montessori',
  cap: 'cap',
  stick: 'stick',
};

function singleCpaSpeechText(meta: SingleCpaObjectMeta): string {
  const nounKey = NOUN_KEY_BY_TYPE[meta.type] ?? meta.subtype;
  const phrase = describeCountedNoun(nounKey, meta.quantity, {
    size: meta.size,
    color: meta.color,
  });
  const qty = numberToSpanishWords(meta.quantity);
  return `${qty} ${phrase}`.trim();
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

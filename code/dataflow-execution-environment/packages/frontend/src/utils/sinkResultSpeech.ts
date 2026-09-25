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
  programError: string | null | undefined,
  _viewMode: ResultViewMode
): string | null {
  // Una salida rota no dice nada; la de al lado sí, aunque hayan fallado en la
  // misma ejecución.
  if ((data.errors?.length ?? 0) > 0) return null;
  if (programError?.trim()) {
    return null;
  }

  const result = data.resultValue;
  if (!result) return null;

  switch (result.kind) {
    case 'boolean':
      return result.value ? 'verdadero' : 'falso';

    case 'numberArray':
      return result.values.map((item) => spokenNumber(item.value)).join(', ');

    case 'number':
      return Number.isFinite(result.value) ? spokenNumber(result.value) : null;

    case 'semantic': {
      if (result.singleCpaObjectMeta) return singleCpaSpeechText(result.singleCpaObjectMeta);

      const description = result.result.description.trim();
      return description ? replaceDigitsWithSpanishWords(description) : null;
    }
  }
}

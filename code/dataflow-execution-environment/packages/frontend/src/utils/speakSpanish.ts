import { replaceDigitsWithSpanishWords } from './spanishNumberWords';

function pickSpanishVoice(): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  return (
    voices.find((v) => v.lang === 'es-ES' || v.lang === 'es_ES') ??
    voices.find((v) => v.lang.startsWith('es-')) ??
    voices.find((v) => v.lang.startsWith('es'))
  );
}

function applySpanishVoice(utterance: SpeechSynthesisUtterance): void {
  const voice = pickSpanishVoice();
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
    return;
  }
  utterance.lang = 'es-ES';
}

/** Reproduce texto en español (voz es-* y dígitos → palabras). */
export function speakSpanish(text: string): void {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  const trimmed = replaceDigitsWithSpanishWords(text.trim());
  if (!trimmed) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(trimmed);
  applySpanishVoice(utterance);

  const voices = window.speechSynthesis.getVoices();
  if (voices.length > 0) {
    window.speechSynthesis.speak(utterance);
    return;
  }

  const onVoices = () => {
    applySpanishVoice(utterance);
    window.speechSynthesis.speak(utterance);
    window.speechSynthesis.onvoiceschanged = null;
  };
  window.speechSynthesis.onvoiceschanged = onVoices;
}

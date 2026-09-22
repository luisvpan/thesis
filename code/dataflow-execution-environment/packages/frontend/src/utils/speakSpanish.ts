import { predict } from '@mintplex-labs/piper-tts-web';
import { replaceDigitsWithSpanishWords } from './spanishNumberWords';

/** Voz neuronal Piper (español de España, calidad media) — no depende de las voces instaladas en el sistema. */
const VOICE_ID = 'es_ES-davefx-medium';

export type SpeechStatus = 'idle' | 'loading' | 'speaking' | 'error';

// Los clics que llegan por el WebSocket de touch (mesa táctil / cámara) se simulan
// con `element.click()`, que el navegador marca como NO confiable (`isTrusted: false`).
// Los navegadores bloquean audio.play()/speechSynthesis.speak() sin un gesto confiable.
// Truco estándar: reutilizar el MISMO <audio> que ya sonó una vez gracias a un gesto
// real (mouse/touch/teclado físico) — una vez desbloqueado, ese elemento puede seguir
// reproduciéndose después aunque la siguiente llamada venga de un clic simulado.
let sharedAudio: HTMLAudioElement | null = null;
let unlocked = false;

function getSharedAudio(): HTMLAudioElement {
  if (!sharedAudio) {
    sharedAudio = new Audio();
  }
  return sharedAudio;
}

/**
 * Desbloquea la reproducción de audio/voz para el resto de la sesión. Debe
 * llamarse desde un manejador de un evento de usuario confiable (real), p. ej.
 * el primer click/touch/tecla física de la página — nunca desde un clic
 * simulado (`element.click()`), que no cuenta para la política de autoplay.
 */
export function unlockSpeechAudio(): void {
  if (unlocked || typeof window === 'undefined') return;
  unlocked = true;

  const audio = getSharedAudio();
  audio.muted = true;
  audio
    .play()
    .then(() => {
      audio.pause();
      audio.muted = false;
    })
    .catch(() => {
      audio.muted = false;
    });

  if (window.speechSynthesis) {
    const warmup = new SpeechSynthesisUtterance(' ');
    warmup.volume = 0;
    window.speechSynthesis.speak(warmup);
  }
}

function stopCurrentAudio(): void {
  const audio = getSharedAudio();
  audio.pause();
  audio.removeAttribute('src');
}

function pickSpanishBrowserVoice(): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  return (
    voices.find((v) => v.lang === 'es-ES' || v.lang === 'es_ES') ??
    voices.find((v) => v.lang.startsWith('es-')) ??
    voices.find((v) => v.lang.startsWith('es'))
  );
}

/** Respaldo si la voz neuronal no pudo cargar (sin internet, WASM bloqueado, etc.). */
function speakWithBrowserFallback(text: string): void {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickSpanishBrowserVoice();
  utterance.lang = voice?.lang ?? 'es-ES';
  if (voice) utterance.voice = voice;
  window.speechSynthesis.speak(utterance);
}

/**
 * Reproduce texto en español con una voz neuronal (Piper) que siempre habla
 * en español, sin depender de qué voces tenga instaladas el sistema operativo.
 * La primera vez descarga el modelo de voz (se cachea en el navegador para
 * las siguientes veces). `onStatusChange` permite mostrar un indicador de
 * carga mientras esto ocurre.
 */
export async function speakSpanish(
  text: string,
  onStatusChange?: (status: SpeechStatus) => void
): Promise<void> {
  const trimmed = replaceDigitsWithSpanishWords(text.trim());
  if (!trimmed) return;

  stopCurrentAudio();
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  onStatusChange?.('loading');

  try {
    const wav = await predict({ text: trimmed, voiceId: VOICE_ID });
    const audio = getSharedAudio();
    audio.muted = false;
    audio.src = URL.createObjectURL(wav);

    audio.onended = () => onStatusChange?.('idle');
    audio.onerror = () => onStatusChange?.('idle');

    onStatusChange?.('speaking');
    await audio.play();
  } catch (err) {
    console.error('No se pudo generar voz neuronal en español, usando voz del navegador', err);
    onStatusChange?.('error');
    speakWithBrowserFallback(trimmed);
    onStatusChange?.('idle');
  }
}

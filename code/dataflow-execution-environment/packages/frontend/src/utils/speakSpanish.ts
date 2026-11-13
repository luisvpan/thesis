import { predict } from '@mintplex-labs/piper-tts-web';
import { replaceDigitsWithSpanishWords } from './spanishNumberWords';

/** Voz neuronal Piper (español de España, calidad media) — no depende de las voces instaladas en el sistema. */
const VOICE_ID = 'es_ES-davefx-medium';

export type SpeechStatus = 'idle' | 'loading' | 'speaking' | 'error';

// ---------------------------------------------------------------------------
// Por qué Web Audio y no un <audio>:
//
// 1. Muchas salidas de audio (HDMI, USB, Bluetooth, algunos drivers de Windows)
//    se "duermen" tras unos segundos de silencio y tardan 200–500 ms en
//    despertar: con un <audio> que arranca en seco, se perdían las primeras
//    sílabas. Aquí mantenemos un AudioContext SIEMPRE corriendo (emitiendo
//    silencio), así el dispositivo nunca se duerme.
// 2. Programamos la voz con un pequeño margen y rampas de volumen de entrada y
//    salida, para evitar chasquidos y que no se coma el final.
// 3. Los clics que llegan por el WebSocket de la mesa táctil se simulan con
//    `element.click()` (`isTrusted: false`) y no cuentan como gesto de usuario.
//    Un AudioContext que se reanudó UNA vez con un gesto real sigue sirviendo
//    para el resto de la sesión, aunque las llamadas siguientes vengan de
//    clics simulados.
// ---------------------------------------------------------------------------

/**
 * Silencio antes de la voz (s). Los parlantes Bluetooth agregan latencia y a
 * veces tardan en "abrir" el audio; con este margen no se come la primera sílaba.
 */
const LEAD_IN_S = 0.35;
/**
 * Amplitud del ruido de mantenimiento (~-66 dBFS). Debe ser lo bastante alto
 * para que el códec Bluetooth (SBC/AAC) NO lo codifique como silencio — si no,
 * el parlante se "duerme" igual — pero inaudible a volumen normal.
 */
const KEEP_ALIVE_AMPLITUDE = 5e-4;
/** Duración de las rampas de volumen de entrada/salida (s), evita "clics". */
const FADE_S = 0.015;
/** Máximo de frases sintetizadas guardadas en memoria. */
const CACHE_LIMIT = 24;

let ctx: AudioContext | null = null;
let masterGain: GainNode | null = null;
let currentSource: AudioBufferSourceNode | null = null;
let currentGain: GainNode | null = null;
/** Se incrementa en cada petición; descarta síntesis que terminan tarde. */
let requestToken = 0;
const bufferCache = new Map<string, AudioBuffer>();

// --- Aviso global de "la voz está hablando" (para bajar la música de fondo) ---
type SpeechActiveListener = (active: boolean) => void;
const speechActiveListeners = new Set<SpeechActiveListener>();
let speechActive = false;

function setSpeechActive(active: boolean): void {
  if (speechActive === active) return;
  speechActive = active;
  speechActiveListeners.forEach((l) => l(active));
}

/** Suscribirse a cuándo la voz empieza/termina de hablar. Devuelve la función para desuscribirse. */
export function subscribeSpeechActive(listener: SpeechActiveListener): () => void {
  speechActiveListeners.add(listener);
  listener(speechActive);
  return () => {
    speechActiveListeners.delete(listener);
  };
}

function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (ctx) return ctx;
  const Ctor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;

  ctx = new Ctor({ latencyHint: 'playback' });
  masterGain = ctx.createGain();
  masterGain.gain.value = 1;
  masterGain.connect(ctx.destination);

  // Fuente permanente "casi silencio": mantiene el flujo de salida abierto para
  // que el dispositivo de audio no entre en ahorro de energía. Usamos un ruido
  // inaudible y no ceros exactos, porque los parlantes Bluetooth (y algunos
  // TV/receptores HDMI) detectan "silencio digital" y se apagan igual.
  const noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
  const samples = noise.getChannelData(0);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = (Math.random() * 2 - 1) * KEEP_ALIVE_AMPLITUDE;
  }
  const keepAlive = ctx.createBufferSource();
  keepAlive.buffer = noise;
  keepAlive.loop = true;
  keepAlive.connect(ctx.destination);
  keepAlive.start();

  // Si el navegador suspende el contexto (p. ej. pestaña en segundo plano),
  // intentamos reanudarlo en cuanto se pueda.
  ctx.addEventListener('statechange', () => {
    if (ctx && ctx.state === 'suspended' && document.visibilityState === 'visible') {
      void ctx.resume().catch(() => undefined);
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (ctx && document.visibilityState === 'visible' && ctx.state !== 'running') {
      void ctx.resume().catch(() => undefined);
    }
  });

  return ctx;
}

/**
 * Desbloquea la reproducción de audio/voz para el resto de la sesión. Debe
 * llamarse desde un manejador de un evento de usuario confiable (real), p. ej.
 * el primer click/touch/tecla física de la página — nunca desde un clic
 * simulado (`element.click()`), que no cuenta para la política de autoplay.
 * Es seguro llamarla muchas veces.
 */
export function unlockSpeechAudio(): void {
  const audioCtx = getAudioContext();
  if (audioCtx && audioCtx.state !== 'running') {
    void audioCtx.resume().catch(() => undefined);
  }
}

function stopCurrentPlayback(): void {
  if (currentSource && currentGain && ctx) {
    const now = ctx.currentTime;
    const g = currentGain.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(0, now + FADE_S);
    const src = currentSource;
    src.onended = null;
    try {
      src.stop(now + FADE_S + 0.005);
    } catch {
      /* ya detenida */
    }
  }
  currentSource = null;
  currentGain = null;
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  setSpeechActive(false);
}

async function synthesize(text: string, audioCtx: AudioContext): Promise<AudioBuffer> {
  const cached = bufferCache.get(text);
  if (cached) {
    // Refrescar posición LRU.
    bufferCache.delete(text);
    bufferCache.set(text, cached);
    return cached;
  }
  const wav = await predict({ text, voiceId: VOICE_ID });
  const buffer = await audioCtx.decodeAudioData(await wav.arrayBuffer());
  bufferCache.set(text, buffer);
  if (bufferCache.size > CACHE_LIMIT) {
    const oldest = bufferCache.keys().next().value;
    if (oldest !== undefined) bufferCache.delete(oldest);
  }
  return buffer;
}

function playBuffer(audioCtx: AudioContext, buffer: AudioBuffer): Promise<void> {
  return new Promise((resolve) => {
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    const gain = audioCtx.createGain();
    source.connect(gain);
    gain.connect(masterGain ?? audioCtx.destination);

    const start = audioCtx.currentTime + LEAD_IN_S;
    const end = start + buffer.duration;
    gain.gain.setValueAtTime(0, audioCtx.currentTime);
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(1, start + FADE_S);
    gain.gain.setValueAtTime(1, Math.max(start + FADE_S, end - FADE_S));
    gain.gain.linearRampToValueAtTime(0, end);

    source.onended = () => {
      if (currentSource === source) {
        currentSource = null;
        currentGain = null;
        setSpeechActive(false);
      }
      gain.disconnect();
      resolve();
    };

    currentSource = source;
    currentGain = gain;
    setSpeechActive(true);
    source.start(start);
  });
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
function speakWithBrowserFallback(text: string): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) {
      resolve();
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = pickSpanishBrowserVoice();
    utterance.lang = voice?.lang ?? 'es-ES';
    if (voice) utterance.voice = voice;
    const done = () => {
      setSpeechActive(false);
      resolve();
    };
    utterance.onend = done;
    utterance.onerror = done;
    setSpeechActive(true);
    window.speechSynthesis.speak(utterance);
  });
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

  const token = ++requestToken;
  stopCurrentPlayback();

  const audioCtx = getAudioContext();
  // Si este clic es un gesto real, esto reanuda el contexto; si es simulado y
  // el contexto ya se desbloqueó antes, no hace falta.
  if (audioCtx && audioCtx.state !== 'running') {
    void audioCtx.resume().catch(() => undefined);
  }

  onStatusChange?.('loading');

  try {
    if (!audioCtx) throw new Error('Web Audio no disponible');
    const buffer = await synthesize(trimmed, audioCtx);
    if (token !== requestToken) return; // llegó otra petición mientras sintetizábamos
    if (audioCtx.state !== 'running') {
      // Sin un gesto real previo, resume() puede no resolverse nunca.
      await Promise.race([
        audioCtx.resume(),
        new Promise((resolve) => setTimeout(resolve, 1500)),
      ]);
      if ((audioCtx.state as AudioContextState) !== 'running') {
        throw new Error('AudioContext bloqueado por el navegador');
      }
    }
    onStatusChange?.('speaking');
    await playBuffer(audioCtx, buffer);
    if (token === requestToken) onStatusChange?.('idle');
  } catch (err) {
    if (token !== requestToken) return;
    console.error('No se pudo generar voz neuronal en español, usando voz del navegador', err);
    onStatusChange?.('speaking');
    await speakWithBrowserFallback(trimmed);
    if (token === requestToken) onStatusChange?.('idle');
  }
}

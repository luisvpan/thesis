import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { PLAYLIST, type Track } from '@/data/musicPlaylist';

const MAX_VOLUME_LEVEL = 5;
const DEFAULT_VOLUME_LEVEL = 3;

// El oído percibe el volumen de forma logarítmica, no lineal: con una escala
// lineal (nivel/MAX), el nivel más bajo ya suena "fuerte" porque una ganancia
// de 0.2 apenas baja unos ~14dB. Usamos una curva exponencial (como el
// potenciómetro "audio taper" de un control de volumen físico) para que cada
// nivel se sienta como un paso perceptual parejo, y el nivel 1 suene realmente bajo.
const MIN_GAIN = 0.02;

function levelToGain(level: number): number {
  if (level <= 0) return 0;
  const t = level / MAX_VOLUME_LEVEL;
  return MIN_GAIN * (1 / MIN_GAIN) ** t;
}

type MusicPlayerContextValue = {
  tracks: Track[];
  currentTrack: Track | undefined;
  isPlaying: boolean;
  /** Nivel de volumen de 0 (mínimo) a 5 (máximo, volumen real 1.0). */
  volumeLevel: number;
  muted: boolean;
  togglePlay: () => void;
  toggleMute: () => void;
  next: () => void;
  previous: () => void;
  /** Vuelve a colocar la canción actual desde el inicio. */
  restart: () => void;
  increaseVolume: () => void;
  decreaseVolume: () => void;
};

const MusicPlayerContext = createContext<MusicPlayerContextValue | null>(null);

/**
 * Reproductor de música de fondo global: vive por encima del router para que
 * la canción siga sonando al navegar entre pantallas de la app.
 */
export function MusicPlayerProvider({ children }: { children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  if (!audioRef.current && typeof window !== 'undefined') {
    audioRef.current = new Audio();
  }

  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(DEFAULT_VOLUME_LEVEL);
  const [muted, setMuted] = useState(false);

  const currentTrack = PLAYLIST[currentIndex];

  const next = useCallback(() => {
    setCurrentIndex((i) => (PLAYLIST.length ? (i + 1) % PLAYLIST.length : 0));
  }, []);

  const previous = useCallback(() => {
    setCurrentIndex((i) => (PLAYLIST.length ? (i - 1 + PLAYLIST.length) % PLAYLIST.length : 0));
  }, []);

  // Cargar la pista actual y seguir reproduciendo si ya estaba sonando.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !currentTrack) return;
    audio.src = currentTrack.url;
    if (isPlaying) {
      void audio.play().catch(() => setIsPlaying(false));
    }
    // Solo debe recargar cuando cambia la pista, no en cada cambio de isPlaying.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIndex, currentTrack]);

  // Avanzar automáticamente a la siguiente canción al terminar.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.addEventListener('ended', next);
    return () => audio.removeEventListener('ended', next);
  }, [next]);

  // Volumen/silencio.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = muted ? 0 : levelToGain(volumeLevel);
  }, [volumeLevel, muted]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      void audio
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(false));
    }
  }, [isPlaying]);

  const toggleMute = useCallback(() => setMuted((m) => !m), []);

  const restart = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    void audio
      .play()
      .then(() => setIsPlaying(true))
      .catch(() => setIsPlaying(false));
  }, []);

  const increaseVolume = useCallback(() => {
    setVolumeLevel((v) => Math.min(MAX_VOLUME_LEVEL, v + 1));
  }, []);

  const decreaseVolume = useCallback(() => {
    setVolumeLevel((v) => Math.max(0, v - 1));
  }, []);

  const value = useMemo<MusicPlayerContextValue>(
    () => ({
      tracks: PLAYLIST,
      currentTrack,
      isPlaying,
      volumeLevel,
      muted,
      togglePlay,
      toggleMute,
      next,
      previous,
      restart,
      increaseVolume,
      decreaseVolume,
    }),
    [
      currentTrack,
      isPlaying,
      volumeLevel,
      muted,
      togglePlay,
      toggleMute,
      next,
      previous,
      restart,
      increaseVolume,
      decreaseVolume,
    ]
  );

  return <MusicPlayerContext.Provider value={value}>{children}</MusicPlayerContext.Provider>;
}

export function useMusicPlayer(): MusicPlayerContextValue {
  const ctx = useContext(MusicPlayerContext);
  if (!ctx) {
    throw new Error('useMusicPlayer debe usarse dentro de MusicPlayerProvider');
  }
  return ctx;
}

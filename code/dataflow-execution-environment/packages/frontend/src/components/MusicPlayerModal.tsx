import {
  Minus,
  Pause,
  Play,
  Plus,
  RotateCcw,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react';
import { useMusicPlayer } from '@/contexts/MusicPlayerContext';

const BIG_BTN =
  'flex min-h-[5.5rem] min-w-[5.5rem] items-center justify-center rounded-xl border-2 border-slate-600 bg-slate-700 text-slate-100 shadow-lg transition-colors hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-40';

type MusicPlayerModalProps = {
  onClose: () => void;
};

export function MusicPlayerModal({ onClose }: MusicPlayerModalProps) {
  const {
    tracks,
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
  } = useMusicPlayer();

  return (
    <div className="fixed inset-0 z-200 flex items-start justify-end pt-28 pr-4" onClick={onClose}>
      <div
        className="w-full max-w-lg rounded-2xl border-2 border-slate-600 bg-slate-800 p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              {tracks.length > 0 ? 'Reproduciendo' : 'Música'}
            </p>
            <h2 className="truncate text-2xl font-bold text-white">
              {currentTrack ? currentTrack.title : 'Sin canciones disponibles'}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className={`${BIG_BTN} min-h-[3.5rem] min-w-[3.5rem] shrink-0`}
            title="Cerrar"
          >
            <X className="h-8 w-8 pointer-events-none" strokeWidth={2.5} aria-hidden />
          </button>
        </div>

        {/* Fila 1: izquierda, play, derecha */}
        <div className="mb-4 flex items-center justify-center gap-3">
          <button
            type="button"
            onClick={previous}
            disabled={tracks.length === 0}
            className={BIG_BTN}
            title="Canción anterior"
          >
            <SkipBack className="h-10 w-10 pointer-events-none" strokeWidth={2} aria-hidden />
          </button>
          <button
            type="button"
            onClick={togglePlay}
            disabled={tracks.length === 0}
            className={`${BIG_BTN} min-h-[7rem] min-w-[7rem] border-teal-500 bg-teal-800 hover:bg-teal-700`}
            title={isPlaying ? 'Pausar' : 'Reproducir'}
          >
            {isPlaying ? (
              <Pause className="h-14 w-14 pointer-events-none" strokeWidth={2} aria-hidden />
            ) : (
              <Play className="h-14 w-14 pointer-events-none" strokeWidth={2} aria-hidden />
            )}
          </button>
          <button
            type="button"
            onClick={next}
            disabled={tracks.length === 0}
            className={BIG_BTN}
            title="Siguiente canción"
          >
            <SkipForward className="h-10 w-10 pointer-events-none" strokeWidth={2} aria-hidden />
          </button>
        </div>

        {/* Fila 2: repetir, mute, control de volumen */}
        <div className="flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            onClick={restart}
            disabled={tracks.length === 0}
            className={BIG_BTN}
            title="Reiniciar canción"
          >
            <RotateCcw className="h-10 w-10 pointer-events-none" strokeWidth={2} aria-hidden />
          </button>
          <button
            type="button"
            onClick={toggleMute}
            className={`${BIG_BTN} ${muted ? 'border-red-500 bg-red-900/60 hover:bg-red-900/80' : ''}`}
            title={muted ? 'Activar sonido' : 'Silenciar'}
          >
            {muted ? (
              <VolumeX className="h-10 w-10 pointer-events-none" strokeWidth={2} aria-hidden />
            ) : (
              <Volume2 className="h-10 w-10 pointer-events-none" strokeWidth={2} aria-hidden />
            )}
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={decreaseVolume}
              disabled={volumeLevel <= 0}
              className={BIG_BTN}
              title="Bajar volumen"
            >
              <Minus className="h-10 w-10 pointer-events-none" strokeWidth={2.5} aria-hidden />
            </button>
            <div className="flex min-w-20 flex-col items-center">
              <span className="text-4xl font-black tabular-nums text-white">{volumeLevel}</span>
              <span className="text-sm font-semibold text-slate-400">de 5</span>
            </div>
            <button
              type="button"
              onClick={increaseVolume}
              disabled={volumeLevel >= 5}
              className={BIG_BTN}
              title="Subir volumen"
            >
              <Plus className="h-10 w-10 pointer-events-none" strokeWidth={2.5} aria-hidden />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

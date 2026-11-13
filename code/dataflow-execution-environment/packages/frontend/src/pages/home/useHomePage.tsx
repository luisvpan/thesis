import { useState } from 'react';

export const homeContainerMotion = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12, delayChildren: 0.1 },
  },
};

export const homeItemMotion = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};

/** Cubos de neón sobre el título (color del borde y del brillo). */
export const homeCubes = [
  { id: 'yellow', size: 'h-20 w-20', color: '#facc15', rotate: -8, y: 12 },
  { id: 'blue', size: 'h-24 w-24', color: '#38bdf8', rotate: 0, y: 0 },
  { id: 'pink', size: 'h-20 w-20', color: '#fb7185', rotate: 8, y: 12 },
] as const;

export const howToPlaySteps = [
  'Coloca tus cubos y cartas sobre la mesa: aparecerán en la pantalla.',
  'Junta números con una carta de suma o resta para formar una operación.',
  'Cambia entre Concreto, Pictórico y Abstracto para ver los números de otra forma.',
  'Toca el ojo para ver como los datos fluyen.',
] as const;

export function useHomePage() {
  const [showMusicModal, setShowMusicModal] = useState(false);
  const [showHowToPlay, setShowHowToPlay] = useState(false);

  return {
    containerMotion: homeContainerMotion,
    itemMotion: homeItemMotion,
    showMusicModal,
    openMusicModal: () => setShowMusicModal(true),
    closeMusicModal: () => setShowMusicModal(false),
    showHowToPlay,
    openHowToPlay: () => setShowHowToPlay(true),
    closeHowToPlay: () => setShowHowToPlay(false),
  };
}

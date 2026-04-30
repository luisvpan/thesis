import { Bug, Gamepad2, Settings } from 'lucide-react';

export const homeContainerMotion = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.2 },
  },
};

export const homeItemMotion = {
  hidden: { opacity: 0, y: 32 },
  show: { opacity: 1, y: 0 },
};

const SECTIONS = [
  {
    id: 'juego',
    title: 'Juego',
    description: 'Mundos 1, 2 y 3 con niveles · Sandbox',
    to: '/juego',
    icon: Gamepad2,
    gradient: 'from-teal-500 to-cyan-600',
    border: 'border-teal-400/30',
    shadow: 'shadow-teal-500/25',
  },
  {
    id: 'configuracion',
    title: 'Configuración',
    description: 'Desbloquear niveles y borrar datos',
    to: '/configuracion',
    icon: Settings,
    gradient: 'from-amber-500 to-orange-600',
    border: 'border-amber-400/30',
    shadow: 'shadow-amber-500/25',
  },
  {
    id: 'dev-mode',
    title: 'Modo Dev',
    description: 'Herramientas de desarrollo para poblar el IDE',
    to: '/dev',
    icon: Bug,
    gradient: 'from-violet-500 to-fuchsia-600',
    border: 'border-violet-400/30',
    shadow: 'shadow-violet-500/25',
  },
] as const;

export function useHomePage() {
  return {
    sections: SECTIONS,
    containerMotion: homeContainerMotion,
    itemMotion: homeItemMotion,
  };
}

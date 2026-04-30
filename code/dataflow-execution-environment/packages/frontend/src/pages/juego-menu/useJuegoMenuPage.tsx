import { FlaskConical, Globe } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const WORLDS = [
  { id: '1', name: 'Mundo 1', icon: Globe },
  { id: '2', name: 'Mundo 2', icon: Globe },
  { id: '3', name: 'Mundo 3', icon: Globe },
  { id: 'sandbox', name: 'Sandbox', icon: FlaskConical },
] as const;

export const juegoMenuContainerMotion = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

export const juegoMenuItemMotion = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};

export function useJuegoMenuPage() {
  const navigate = useNavigate();

  return {
    worlds: WORLDS,
    goBack: () => navigate('/'),
    containerMotion: juegoMenuContainerMotion,
    itemMotion: juegoMenuItemMotion,
  };
}

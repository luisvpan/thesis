import { LockOpen, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const OPTIONS = [
  { id: 'unlock', name: 'Desbloquear Niveles', icon: LockOpen },
  { id: 'clear', name: 'Borrar Datos', icon: Trash2 },
] as const;

export const configuracionContainerMotion = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.1 },
  },
};

export const configuracionItemMotion = {
  hidden: { opacity: 0, x: -20 },
  show: { opacity: 1, x: 0 },
};

export function useConfiguracionPage() {
  const navigate = useNavigate();
  const [unlocked, setUnlocked] = useState(false);
  const [cleared, setCleared] = useState(false);

  const handleUnlock = () => {
    setUnlocked(true);
    setTimeout(() => setUnlocked(false), 2000);
  };

  const handleClear = () => {
    if (window.confirm('¿Borrar todos los datos guardados? Esta acción no se puede deshacer.')) {
      setCleared(true);
      setTimeout(() => setCleared(false), 2000);
    }
  };

  return {
    options: OPTIONS,
    unlocked,
    cleared,
    handleUnlock,
    handleClear,
    goBack: () => navigate('/'),
    containerMotion: configuracionContainerMotion,
    itemMotion: configuracionItemMotion,
  };
}

import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

const LEVELS = [1, 2, 3, 4] as const;
const WORLD_NAMES: Record<string, string> = {
  '1': 'Mundo 1',
  '2': 'Mundo 2',
  '3': 'Mundo 3',
  sandbox: 'Sandbox',
};

export const worldLevelsContainerMotion = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.15 },
  },
};

export const worldLevelsItemMotion = {
  hidden: { opacity: 0, scale: 0.9 },
  show: { opacity: 1, scale: 1 },
};

export function useWorldLevelsPage() {
  const { worldId } = useParams<{ worldId: string }>();
  const navigate = useNavigate();

  useEffect(() => {
    if (!worldId) {
      navigate('/juego', { replace: true });
      return;
    }

    if (worldId === 'sandbox') {
      navigate('/ide/sandbox', { replace: true });
    }
  }, [navigate, worldId]);

  return {
    worldId,
    worldName: worldId ? WORLD_NAMES[worldId] ?? `Mundo ${worldId}` : '',
    levels: LEVELS,
    goBack: () => navigate('/juego'),
    containerMotion: worldLevelsContainerMotion,
    itemMotion: worldLevelsItemMotion,
  };
}

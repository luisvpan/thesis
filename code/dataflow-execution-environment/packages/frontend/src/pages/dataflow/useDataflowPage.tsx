import { useRef } from 'react';
import { useParams } from 'react-router-dom';
import { getLevelConfig } from '@/data/levelConfig';

export function useDataflowPage(isSandbox: boolean) {
  const params = useParams();
  const worldId = params.worldId;
  const level = params.level;
  const levelConfig = getLevelConfig(worldId, level, isSandbox);
  const backTo = worldId ? (isSandbox ? '/juego' : `/juego/${worldId}`) : '/';
  const flowContainerRef = useRef<HTMLDivElement>(null);

  return {
    levelConfig,
    backTo,
    flowContainerRef,
  };
}

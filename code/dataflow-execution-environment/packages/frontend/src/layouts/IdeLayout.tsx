import { Outlet } from 'react-router-dom';
import { VisionProvider } from '@/contexts/VisionContext';

/**
 * Solo en rutas `/ide/*`: WebSocket de visión y estado (evita conexión fuera del IDE).
 */
export function IdeLayout() {
  return (
    <VisionProvider>
      <Outlet />
    </VisionProvider>
  );
}

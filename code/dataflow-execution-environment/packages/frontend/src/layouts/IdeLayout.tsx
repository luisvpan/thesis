import { Outlet } from 'react-router-dom';
import { VisionProvider } from '@/contexts/VisionContext';

/**
 * Solo en rutas `/ide/*`: WebSocket de visión (evita conexión fuera del IDE).
 * Touch se maneja globalmente en main.tsx.
 */
export function IdeLayout() {
  return (
    <VisionProvider>
      <Outlet />
    </VisionProvider>
  );
}

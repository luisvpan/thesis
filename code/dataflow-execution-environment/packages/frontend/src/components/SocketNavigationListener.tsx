import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSocket } from '@/contexts/SocketContext';
import type { SocketNavigatePayload, NavigateTo } from '@/types/socket-types';

const ROUTES: Record<Exclude<NavigateTo, 'atras' | 'niveles'>, string> = {
  menu: '/',
  configuracion: '/configuracion',
  jugar: '/juego',
  mundos: '/juego',
};

/**
 * Escucha el evento "navigate" del backend y navega en la app (para uso sin mouse).
 * El backend puede enviar:
 * - { path: "/ruta" } para ir a una ruta directa
 * - { to: "menu" | "configuracion" | "atras" | "jugar" | "mundos" | "niveles", worldId?: "1"|"2"|"3" }
 */
export function SocketNavigationListener() {
  const socket = useSocket();
  const navigate = useNavigate();

  useEffect(() => {
    if (!socket) return;

    const handleNavigate = (payload: SocketNavigatePayload) => {
      if (payload.path) {
        navigate(payload.path);
        return;
      }
      const to = payload.to;
      if (!to) return;

      if (to === 'atras') {
        navigate(-1);
        return;
      }
      if (to === 'niveles' && payload.worldId) {
        navigate(`/juego/${payload.worldId}`);
        return;
      }
      const path = ROUTES[to as keyof typeof ROUTES];
      if (path) navigate(path);
    };

    socket.on('navigate', handleNavigate);
    return () => {
      socket.off('navigate', handleNavigate);
    };
  }, [socket, navigate]);

  return null;
}

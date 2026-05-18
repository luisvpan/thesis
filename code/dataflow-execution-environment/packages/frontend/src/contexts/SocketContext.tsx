import {
  createContext,
  useContext,
  useEffect,
  useState,
  useMemo,
  type ReactNode,
} from 'react';
import { io, type Socket } from 'socket.io-client';
import { logger } from '@/lib/logger';

const SOCKET_URL =
  typeof import.meta.env.VITE_SOCKET_URL === 'string' && import.meta.env.VITE_SOCKET_URL
    ? import.meta.env.VITE_SOCKET_URL
    : window.location.origin;

const SocketContext = createContext<Socket | null>(null);

export function SocketProvider({ children }: { children: ReactNode }) {
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    const s = io(SOCKET_URL, {
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    s.on('connect', () => {
      logger.socket.debug('Connected', { id: s.id });
    });

    /** Log de todo evento que emita el servidor Socket.IO (addNode, navigate, etc.) */
    s.onAny((event, ...args) => {
      logger.socketBackend.debug('Event received', { event, args });
    });
    s.on('disconnect', (reason) => {
      logger.socket.debug('Disconnected', { reason });
    });
    s.on('connect_error', (err) => {
      logger.socket.warn('Connection error', { message: err.message });
    });

    setSocket(s);
    return () => {
      s.removeAllListeners();
      s.close();
      setSocket(null);
    };
  }, []);

  const value = useMemo(() => socket, [socket]);
  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
}

export function useSocket(): Socket | null {
  return useContext(SocketContext);
}

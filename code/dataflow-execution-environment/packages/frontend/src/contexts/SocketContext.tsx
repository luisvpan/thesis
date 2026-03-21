import {
  createContext,
  useContext,
  useEffect,
  useState,
  useMemo,
  type ReactNode,
} from 'react';
import { io, type Socket } from 'socket.io-client';

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
      console.debug('[Socket] connected', s.id);
    });

    /** Log de todo evento que emita el servidor Socket.IO (addNode, navigate, etc.) */
    s.onAny((event, ...args) => {
      console.log('[backend:socket.io]', event, args);
    });
    s.on('disconnect', (reason) => {
      console.debug('[Socket] disconnected', reason);
    });
    s.on('connect_error', (err) => {
      console.warn('[Socket] connect_error', err.message);
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

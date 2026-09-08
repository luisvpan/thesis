import { useNavigate } from 'react-router-dom';

const DEV_ENTRIES = [
  {
    id: 'sandbox',
    title: 'Sandbox (Dev)',
    description: 'Abrir IDE sandbox con herramientas de modo dev',
    to: '/ide/sandbox?dev=1',
  },
  {
    id: 'world-1-level-1',
    title: 'Mundo 1 · Nivel 1 (Dev)',
    description: 'Abrir nivel con detección normal y utilidades dev',
    to: '/ide/1/1?dev=1',
  },
  {
    id: 'world-2-level-1',
    title: 'Mundo 2 · Nivel 1 (Dev)',
    description: 'Ruta rápida para probar operadores y resultados',
    to: '/ide/2/1?dev=1',
  },
  {
    id: 'world-3-level-1',
    title: 'Mundo 3 · Nivel 1 (Dev)',
    description: 'Ruta rápida para pruebas de integración visual',
    to: '/ide/3/1?dev=1',
  },
] as const;

export function useDevModePage() {
  const navigate = useNavigate();
  return {
    entries: DEV_ENTRIES,
    goBack: () => navigate('/'),
  };
}

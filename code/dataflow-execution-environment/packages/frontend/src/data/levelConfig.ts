import type { MathOperatorType } from '@/types/card-types';

export interface LevelConfig {
  title: string;
  subtitle: string;
  rule: string;
  /** Números que el niño puede usar en este nivel (cartas en la mochila). */
  numbers: number[];
  /** Operadores que puede usar en este nivel. */
  operators: MathOperatorType[];
}

type WorldId = '1' | '2' | '3';
type LevelId = '1' | '2' | '3' | '4';

const LEVELS: Record<WorldId, Record<LevelId, LevelConfig>> = {
  '1': {
    '1': {
      title: 'Mundo 1 · Nivel 1',
      subtitle: 'Suma dos números del 0 al 5',
      rule: 'Usa solo los números que aparecen en la mochila. Conecta dos números a las entradas del operador de suma (entrada A y entrada B). El resultado aparecerá debajo del operador.',
      numbers: [0, 1, 2, 3, 4, 5],
      operators: ['adicion'],
    },
    '2': {
      title: 'Mundo 1 · Nivel 2',
      subtitle: 'Suma con números del 0 al 9',
      rule: 'Puedes usar cualquier número del 0 al 9. Arrastra dos números al operador de suma y conecta cada uno a una entrada (A o B). El orden no cambia el resultado en la suma.',
      numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      operators: ['adicion'],
    },
    '3': {
      title: 'Mundo 1 · Nivel 3',
      subtitle: 'Introduce la resta',
      rule: 'En la resta, el orden sí importa: el número en la entrada A menos el de la entrada B. Usa los números permitidos y el operador de resta para obtener el resultado.',
      numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      operators: ['sustraccion'],
    },
    '4': {
      title: 'Mundo 1 · Nivel 4',
      subtitle: 'Sumas y restas',
      rule: 'Puedes usar suma y resta. Conecta números a los operadores. El resultado de un operador puede conectarse como entrada de otro para hacer cadenas (por ejemplo: 5 − 2 = 3, y luego 3 + 4 = 7).',
      numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      operators: ['adicion', 'sustraccion'],
    },
  },
  '2': {
    '1': {
      title: 'Mundo 2 · Nivel 1',
      subtitle: 'Sumas con dígitos pares',
      rule: 'En este nivel solo puedes usar números pares: 0, 2, 4, 6, 8. Elige dos de ellos y conéctalos al operador de suma.',
      numbers: [0, 2, 4, 6, 8],
      operators: ['adicion'],
    },
    '2': {
      title: 'Mundo 2 · Nivel 2',
      subtitle: 'Sumas con dígitos impares',
      rule: 'Solo puedes usar números impares: 1, 3, 5, 7, 9. Conecta dos de ellos al operador de suma y observa el resultado.',
      numbers: [1, 3, 5, 7, 9],
      operators: ['adicion'],
    },
    '3': {
      title: 'Mundo 2 · Nivel 3',
      subtitle: 'Restas con números pequeños',
      rule: 'Usa solo los números del 0 al 5. Practica restas donde el primer número (entrada A) sea mayor o igual que el segundo (entrada B) para obtener un resultado positivo.',
      numbers: [0, 1, 2, 3, 4, 5],
      operators: ['sustraccion'],
    },
    '4': {
      title: 'Mundo 2 · Nivel 4',
      subtitle: 'Combinando pares e impares',
      rule: 'Tienes todos los números y ambos operadores. Construye una cadena: por ejemplo suma dos números y luego resta otro al resultado, o haz dos sumas seguidas.',
      numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      operators: ['adicion', 'sustraccion'],
    },
  },
  '3': {
    '1': {
      title: 'Mundo 3 · Nivel 1',
      subtitle: 'Solo números del 1 al 4',
      rule: 'Usa únicamente 1, 2, 3 y 4. Conecta dos de estos números al operador de suma. ¿Cuántas sumas distintas puedes hacer?',
      numbers: [1, 2, 3, 4],
      operators: ['adicion'],
    },
    '2': {
      title: 'Mundo 3 · Nivel 2',
      subtitle: 'Restas con 5, 6, 7, 8 y 9',
      rule: 'Solo los números 5, 6, 7, 8 y 9. Usa el operador de resta. Recuerda: conecta primero el número mayor a la entrada A y el menor a la entrada B si quieres un resultado positivo.',
      numbers: [5, 6, 7, 8, 9],
      operators: ['sustraccion'],
    },
    '3': {
      title: 'Mundo 3 · Nivel 3',
      subtitle: 'Tres números en cadena',
      rule: 'Usa suma y resta. El objetivo es usar al menos tres números en cadena: por ejemplo (3 + 5) − 2, o 9 − 4 + 1. Conecta la salida de un operador a la entrada de otro.',
      numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      operators: ['adicion', 'sustraccion'],
    },
    '4': {
      title: 'Mundo 3 · Nivel 4',
      subtitle: 'Libertad total',
      rule: 'Tienes todas las cartas. Construye el flujo que quieras: varias sumas, varias restas o mezclas. El resultado final se mostrará en el pie de pantalla al pulsar Ejecutar.',
      numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      operators: ['adicion', 'sustraccion'],
    },
  },
};

/** Config por defecto para Sandbox (todas las cartas). */
export const SANDBOX_CONFIG: LevelConfig = {
  title: 'Sandbox',
  subtitle: 'Modo libre: explora sumas y restas',
  rule: 'En el sandbox no hay restricciones. Usa el canvas para explorar; las cartas se agregan desde el backend por socket.',
  numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  operators: ['adicion', 'sustraccion'],
};

/**
 * Obtiene la configuración del nivel (título, subtítulo, regla y mochila) para el IDE.
 * Si es sandbox o no hay mundo/nivel válido, devuelve SANDBOX_CONFIG.
 */
export function getLevelConfig(
  worldId: string | undefined,
  level: string | undefined,
  isSandbox: boolean
): LevelConfig {
  if (isSandbox || !worldId || !level) {
    return SANDBOX_CONFIG;
  }
  const world = LEVELS[worldId as WorldId];
  const levelConfig = world?.[level as LevelId];
  return levelConfig ?? SANDBOX_CONFIG;
}

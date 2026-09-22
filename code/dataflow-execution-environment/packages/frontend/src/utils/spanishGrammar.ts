/**
 * Diccionario gramatical centralizado para construir frases en español que
 * concuerdan en género y número (p. ej. "3 manzanas rojas", no "3 manzanas rojo").
 *
 * Usado por executeProgram.ts (descripción textual), ProgramOutputFlowNode.tsx
 * (encabezado de resultado único) y sinkResultSpeech.ts (texto para TTS) para
 * evitar que cada uno concatene palabras crudas por su cuenta.
 */

export type Gender = 'm' | 'f';

interface NounInfo {
  singular: string;
  plural: string;
  gender: Gender;
}

/**
 * Sustantivos conocidos por tipo/subtipo interno del intérprete. Incluye alias
 * para identificadores que el intérprete ya entrega en plural (p. ej. "peras",
 * "uvas") para que no se les agregue una "s" extra.
 */
const NOUNS: Record<string, NounInfo> = {
  // Formas geométricas
  triangulo: { singular: 'triángulo', plural: 'triángulos', gender: 'm' },
  cuadrado: { singular: 'cuadrado', plural: 'cuadrados', gender: 'm' },
  circulo: { singular: 'círculo', plural: 'círculos', gender: 'm' },
  rectangulo: { singular: 'rectángulo', plural: 'rectángulos', gender: 'm' },
  rombo: { singular: 'rombo', plural: 'rombos', gender: 'm' },
  estrella: { singular: 'estrella', plural: 'estrellas', gender: 'f' },
  trapecio: { singular: 'trapecio', plural: 'trapecios', gender: 'm' },
  forma: { singular: 'forma', plural: 'formas', gender: 'f' },
  // Comida
  manzana: { singular: 'manzana', plural: 'manzanas', gender: 'f' },
  hamburguesa: { singular: 'hamburguesa', plural: 'hamburguesas', gender: 'f' },
  pasta: { singular: 'pasta', plural: 'pastas', gender: 'f' },
  naranja: { singular: 'naranja', plural: 'naranjas', gender: 'f' },
  pera: { singular: 'pera', plural: 'peras', gender: 'f' },
  peras: { singular: 'pera', plural: 'peras', gender: 'f' },
  uva: { singular: 'uva', plural: 'uvas', gender: 'f' },
  uvas: { singular: 'uva', plural: 'uvas', gender: 'f' },
  comida: { singular: 'comida', plural: 'comidas', gender: 'f' },
  // Material concreto (el subtipo real que llega del intérprete es el color;
  // el sustantivo se deriva del tipo, ver `subtypeNounKey` en executeProgram.ts)
  montessori: { singular: 'cubo', plural: 'cubos', gender: 'm' },
  cap: { singular: 'tapa', plural: 'tapas', gender: 'f' },
  stick: { singular: 'paleta', plural: 'paletas', gender: 'f' },
  // Abstracto
  numero: { singular: 'número', plural: 'números', gender: 'm' },
};

function nounInfo(key: string): NounInfo {
  return NOUNS[key] ?? { singular: key, plural: `${key}s`, gender: 'm' };
}

export function nounGender(key: string): Gender {
  return nounInfo(key).gender;
}

/** Forma singular o plural correcta del sustantivo (p. ej. "triángulo"/"triángulos"). */
export function nounForm(key: string, count: number): string {
  const info = nounInfo(key);
  return count === 1 ? info.singular : info.plural;
}

interface AdjectiveForms {
  ms: string;
  fs: string;
  mp: string;
  fp: string;
}

// Colores: ShapeColor, MontessoriColor, CapColor, StickColor, CarColor
const COLOR_ADJECTIVES: Record<string, AdjectiveForms> = {
  rojo: { ms: 'rojo', fs: 'roja', mp: 'rojos', fp: 'rojas' },
  azul: { ms: 'azul', fs: 'azul', mp: 'azules', fp: 'azules' },
  amarillo: { ms: 'amarillo', fs: 'amarilla', mp: 'amarillos', fp: 'amarillas' },
  verde: { ms: 'verde', fs: 'verde', mp: 'verdes', fp: 'verdes' },
  morado: { ms: 'morado', fs: 'morada', mp: 'morados', fp: 'moradas' },
  naranja: { ms: 'naranja', fs: 'naranja', mp: 'naranjas', fp: 'naranjas' },
  blanco: { ms: 'blanco', fs: 'blanca', mp: 'blancos', fp: 'blancas' },
  negro: { ms: 'negro', fs: 'negra', mp: 'negros', fp: 'negras' },
  gris: { ms: 'gris', fs: 'gris', mp: 'grises', fp: 'grises' },
  madera: { ms: 'de madera', fs: 'de madera', mp: 'de madera', fp: 'de madera' },
  'azul-oscuro': {
    ms: 'azul oscuro',
    fs: 'azul oscuro',
    mp: 'azules oscuros',
    fp: 'azules oscuras',
  },
};

// Tamaños: ShapeSize
const SIZE_ADJECTIVES: Record<string, AdjectiveForms> = {
  pequeño: { ms: 'pequeño', fs: 'pequeña', mp: 'pequeños', fp: 'pequeñas' },
  mediano: { ms: 'mediano', fs: 'mediana', mp: 'medianos', fp: 'medianas' },
  grande: { ms: 'grande', fs: 'grande', mp: 'grandes', fp: 'grandes' },
};

// Alias por si llega una variante ya conjugada desde otra fuente de datos.
const ADJECTIVE_ALIASES: Record<string, string> = {
  pequeña: 'pequeño',
  mediana: 'mediano',
  roja: 'rojo',
  rojos: 'rojo',
  rojas: 'rojo',
  amarilla: 'amarillo',
  amarillos: 'amarillo',
  amarillas: 'amarillo',
  morada: 'morado',
  morados: 'morado',
  moradas: 'morado',
  blanca: 'blanco',
  blancos: 'blanco',
  blancas: 'blanco',
  negra: 'negro',
  negros: 'negro',
  negras: 'negro',
};

function conjugate(
  raw: string,
  dict: Record<string, AdjectiveForms>,
  gender: Gender,
  plural: boolean
): string {
  const key = ADJECTIVE_ALIASES[raw] ?? raw;
  const forms = dict[key];
  if (!forms) return raw; // valor desconocido: se muestra tal cual, sin adivinar
  if (gender === 'f') return plural ? forms.fp : forms.fs;
  return plural ? forms.mp : forms.ms;
}

/** Conjuga un color para que concuerde en género/número con el sustantivo. */
export function conjugateColor(color: string, gender: Gender, count: number): string {
  return conjugate(color, COLOR_ADJECTIVES, gender, count !== 1);
}

/** Conjuga un tamaño para que concuerde en género/número con el sustantivo. */
export function conjugateSize(size: string, gender: Gender, count: number): string {
  return conjugate(size, SIZE_ADJECTIVES, gender, count !== 1);
}

/**
 * Construye una frase gramaticalmente correcta a partir de un sustantivo
 * conocido y sus atributos: "manzanas rojas", "triángulo grande azul",
 * "cubos rojos". El orden tamaño → color sigue el uso habitual en español.
 */
export function describeCountedNoun(
  nounKey: string,
  count: number,
  attrs: { size?: string; color?: string } = {}
): string {
  const gender = nounGender(nounKey);
  const parts = [nounForm(nounKey, count)];
  if (attrs.size) parts.push(conjugateSize(attrs.size, gender, count));
  if (attrs.color) parts.push(conjugateColor(attrs.color, gender, count));
  return parts.join(' ');
}

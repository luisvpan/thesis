const ONES = [
  'cero',
  'uno',
  'dos',
  'tres',
  'cuatro',
  'cinco',
  'seis',
  'siete',
  'ocho',
  'nueve',
] as const;

const TEN_TO_NINETEEN = [
  'diez',
  'once',
  'doce',
  'trece',
  'catorce',
  'quince',
  'dieciseis',
  'diecisiete',
  'dieciocho',
  'diecinueve',
] as const;

const TENS = [
  '',
  '',
  'veinte',
  'treinta',
  'cuarenta',
  'cincuenta',
  'sesenta',
  'setenta',
  'ochenta',
  'noventa',
] as const;

/** Entero 0–99 → palabras en español (sin tilde, mejor para TTS). */
export function numberToSpanishWords(n: number): string {
  if (!Number.isInteger(n) || n < 0 || n > 99) {
    return String(n);
  }
  if (n < 10) return ONES[n];
  if (n < 20) return TEN_TO_NINETEEN[n - 10];
  if (n < 30) {
    return n === 20 ? 'veinte' : `veinti${ONES[n - 20]}`;
  }
  const tens = Math.floor(n / 10);
  const ones = n % 10;
  if (ones === 0) return TENS[tens];
  return `${TENS[tens]} y ${ONES[ones]}`;
}

/** Sustituye secuencias de dígitos por palabras (p. ej. «16 objetos» → «dieciseis objetos»). */
export function replaceDigitsWithSpanishWords(text: string): string {
  return text.replace(/\d+/g, (digits) => {
    const n = Number.parseInt(digits, 10);
    return Number.isNaN(n) ? digits : numberToSpanishWords(n);
  });
}

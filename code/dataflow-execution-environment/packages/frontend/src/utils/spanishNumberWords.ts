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

const HUNDREDS = [
  '',
  'ciento',
  'doscientos',
  'trescientos',
  'cuatrocientos',
  'quinientos',
  'seiscientos',
  'setecientos',
  'ochocientos',
  'novecientos',
] as const;

/** Máximo entero que sabemos decir (999 999 999 999). */
const MAX_SPOKEN = 999_999_999_999;

function under100(n: number): string {
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

function under1000(n: number): string {
  if (n === 100) return 'cien';
  const hundreds = Math.floor(n / 100);
  const rest = n % 100;
  if (hundreds === 0) return under100(rest);
  return rest === 0 ? HUNDREDS[hundreds] : `${HUNDREDS[hundreds]} ${under100(rest)}`;
}

/** «uno» se apocopa delante de «mil»/«millones»: veintiuno → veintiun, treinta y uno → treinta y un. */
function apocopate(words: string): string {
  return words.endsWith('uno') ? `${words.slice(0, -3)}un` : words;
}

function integerToWords(n: number): string {
  if (n < 1000) return under1000(n);

  if (n < 1_000_000) {
    const thousands = Math.floor(n / 1000);
    const rest = n % 1000;
    const head = thousands === 1 ? 'mil' : `${apocopate(under1000(thousands))} mil`;
    return rest === 0 ? head : `${head} ${under1000(rest)}`;
  }

  const millions = Math.floor(n / 1_000_000);
  const rest = n % 1_000_000;
  const head =
    millions === 1 ? 'un millón' : `${apocopate(integerToWords(millions))} millones`;
  return rest === 0 ? head : `${head} ${integerToWords(rest)}`;
}

/**
 * Número → palabras en español (sin tilde en las palabras habituales, mejor
 * para TTS). Soporta enteros hasta 999 999 999 999, negativos («menos …») y
 * decimales (hasta 2 cifras: 2.5 → «dos coma cinco», 3.25 → «tres coma veinticinco»).
 * Si no se puede decir, devuelve los dígitos tal cual.
 */
export function numberToSpanishWords(n: number): string {
  if (!Number.isFinite(n)) return String(n);
  if (n < 0) return `menos ${numberToSpanishWords(-n)}`;

  const rounded = Math.round(n * 100) / 100;
  const intPart = Math.trunc(rounded);
  if (intPart > MAX_SPOKEN) return String(n);

  const intWords = integerToWords(intPart);
  const decimals = Math.round((rounded - intPart) * 100);
  if (decimals === 0) return intWords;

  // 2.5 → «cinco» (no «cincuenta»); 2.05 → «cero cinco».
  const decimalWords =
    decimals % 10 === 0
      ? under100(decimals / 10)
      : decimals < 10
        ? `cero ${ONES[decimals]}`
        : under100(decimals);
  return `${intWords} coma ${decimalWords}`;
}

/** Sustituye números escritos con dígitos por palabras (p. ej. «16 objetos» → «dieciseis objetos»). */
export function replaceDigitsWithSpanishWords(text: string): string {
  return text.replace(/\d+(?:[.,]\d+)?/g, (match) => {
    const n = Number.parseFloat(match.replace(',', '.'));
    return Number.isNaN(n) ? match : numberToSpanishWords(n);
  });
}

/**
 * Mapea nombres de clase YOLO (one, two, … o español) a dígito 1..9 para nodos número.
 */
export function visionLabelToDigit(label: string): number | undefined {
  const n = label.trim().toLowerCase();

  const digitWords: Record<string, number> = {
    one: 1,
    two: 2,
    three: 3,
    four: 4,
    five: 5,
    six: 6,
    seven: 7,
    eight: 8,
    nine: 9,
    uno: 1,
    dos: 2,
    tres: 3,
    cuatro: 4,
    cinco: 5,
    seis: 6,
    siete: 7,
    ocho: 8,
    nueve: 9,
  };

  if (digitWords[n] != null) return digitWords[n];

  const m = /^(\d)$/.exec(n);
  if (m) {
    const v = Number(m[1]);
    if (v >= 1 && v <= 9) return v;
  }

  return undefined;
}

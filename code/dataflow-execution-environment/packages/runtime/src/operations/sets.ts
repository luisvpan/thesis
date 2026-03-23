function deepEquals(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (typeof a !== 'object' || a === null || b === null) return false;

  const objA = a as Record<string, unknown>;
  const objB = b as Record<string, unknown>;

  const keysA = Object.keys(objA);
  const keysB = Object.keys(objB);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (!keysB.includes(key)) return false;
    if (!deepEquals(objA[key], objB[key])) return false;
  }

  return true;
}

export function UNION(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set1, set2] = inputs;
  const elements1 = (set1.value as { kind: string; elements: unknown[] }).elements;
  const elements2 = (set2.value as { kind: string; elements: unknown[] }).elements;
  const result = [...elements1];
  for (const elem of elements2) {
    if (!result.some(e => deepEquals(e, elem))) {
      result.push(elem);
    }
  }
  return { kind: "set", elements: result };
}

export function INTERSECTION(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set1, set2] = inputs;
  const elements1 = (set1.value as { kind: string; elements: unknown[] }).elements;
  const elements2 = (set2.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: elements1.filter(elem => elements2.some(e => deepEquals(e, elem))) };
}

export function DIFFERENCE(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set1, set2] = inputs;
  const elements1 = (set1.value as { kind: string; elements: unknown[] }).elements;
  const elements2 = (set2.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: elements1.filter(elem => !elements2.some(e => deepEquals(e, elem))) };
}

export function COMPLEMENT(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [universe, subset] = inputs;
  const universalElements = (universe.value as { kind: string; elements: unknown[] }).elements;
  const subsetElements = (subset.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: universalElements.filter(elem => !subsetElements.some(e => deepEquals(e, elem))) };
}

export function SORT(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements;
  
  return { kind: "set", elements: [...elements].sort((a, b) => {
    const getNumericValue = (val: unknown): number => {
      if (typeof val === 'number') return val;
      if (typeof val === 'object' && val !== null) {
        const obj = val as Record<string, unknown>;
        if ('kind' in obj) {
          const kind = obj.kind as string;
          if (kind === 'natural' || kind === 'integer' || kind === 'decimal') {
            return obj.value as number;
          }
          if (kind === 'fraction' && 'numerator' in obj && 'denominator' in obj) {
            const frac = obj as { numerator: number; denominator: number };
            return frac.numerator / frac.denominator;
          }
        }
      }
      throw new Error(
        '⚠️ ¡Ups! SORT solo funciona con números (natural, integer, decimal, fraction).\n' +
        'Tu conjunto contiene otro tipo de valor. Por favor, usa solo números en SORT.'
      );
    };
    const aNum = getNumericValue(a);
    const bNum = getNumericValue(b);
    return aNum - bNum;
  })};
}

export function ALPHABETICAL_SORT(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements;
  
  const getTextValue = (val: unknown): string => {
    if (typeof val === 'object' && val !== null) {
      const obj = val as Record<string, unknown>;
      if ('kind' in obj && obj.kind === 'text' && 'value' in obj) {
        return obj.value as string;
      }
    }
    throw new Error(
      '⚠️ ¡Ups! ALPHABETICAL_SORT solo funciona con texto (text).\n' +
      'Tu conjunto contiene otro tipo de valor. Por favor, usa solo texto en ALPHABETICAL_SORT.'
    );
  };
  
  return { kind: "set", elements: [...elements].sort((a, b) => {
    const strA = getTextValue(a);
    const strB = getTextValue(b);
    return strA.localeCompare(strB);
  })};
}

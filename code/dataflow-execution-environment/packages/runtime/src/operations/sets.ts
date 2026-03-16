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
      if (typeof val === 'object' && val !== null && 'value' in val) {
        return (val as { value: number }).value;
      }
      if (typeof val === 'object' && val !== null && 'numerator' in val && 'denominator' in val) {
        const frac = val as { numerator: number; denominator: number };
        return frac.numerator / frac.denominator;
      }
      return 0;
    };
    const aNum = getNumericValue(a);
    const bNum = getNumericValue(b);
    return aNum - bNum;
  })};
}

export function ALPHABETICAL_SORT(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: [...elements].sort((a, b) => {
    const strA = String(a);
    const strB = String(b);
    return strA.localeCompare(strB);
  })};
}

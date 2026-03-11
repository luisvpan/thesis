export function UNION(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set1, set2] = inputs;
  const elements1 = (set1.value as { kind: string; elements: unknown[] }).elements;
  const elements2 = (set2.value as { kind: string; elements: unknown[] }).elements;
  const result = [...elements1];
  for (const elem of elements2) {
    if (!result.includes(elem)) {
      result.push(elem);
    }
  }
  return { kind: "set", elements: result };
}

export function INTERSECTION(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set1, set2] = inputs;
  const elements1 = (set1.value as { kind: string; elements: unknown[] }).elements;
  const elements2 = (set2.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: elements1.filter(elem => elements2.includes(elem)) };
}

export function DIFFERENCE(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set1, set2] = inputs;
  const elements1 = (set1.value as { kind: string; elements: unknown[] }).elements;
  const elements2 = (set2.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: elements1.filter(elem => !elements2.includes(elem)) };
}

export function COMPLEMENT(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [universe, subset] = inputs;
  const universalElements = (universe.value as { kind: string; elements: unknown[] }).elements;
  const subsetElements = (subset.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: universalElements.filter(elem => !subsetElements.includes(elem)) };
}

export function SORT(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements;
  return { kind: "set", elements: [...elements].sort((a, b) => {
    if (typeof a === 'number' && typeof b === 'number') {
      return a - b;
    }
    return 0;
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

/**
 * Post-ordering utilities for numerical sorting of array results.
 * Used when the user selects "numerical" ordering strategy in the UI.
 */

import type { ResultVisualItem } from '@/services/executeProgram';

type RuntimeElement = Record<string, unknown>;

// ============================================================================
// Helpers for extracting information from RuntimeValue elements
// ============================================================================

function getQuantity(elem: unknown): number {
  if (!elem || typeof elem !== 'object') return 0;
  const obj = elem as RuntimeElement;

  if (obj.kind === 'racional' && obj.value) {
    return Number((obj.value as { valueOf(): number }).valueOf());
  }
  if (obj.kind === 'cpa' && obj.quantity) {
    return Number((obj.quantity as { valueOf(): number }).valueOf());
  }
  return 0;
}

function getCategory(elem: unknown): 'abstracto' | 'pictorico' | 'concreto' {
  if (!elem || typeof elem !== 'object') return 'abstracto';
  const obj = elem as RuntimeElement;

  if (obj.kind === 'racional') return 'abstracto';

  if (obj.kind === 'cpa' && obj.category) {
    return obj.category as 'abstracto' | 'pictorico' | 'concreto';
  }

  return 'abstracto';
}

function getType(elem: unknown): string {
  if (!elem || typeof elem !== 'object') return 'racional';
  const obj = elem as RuntimeElement;

  if (obj.kind === 'racional') return 'racional';

  if (obj.kind === 'cpa' && obj.type) {
    return obj.type as string;
  }

  return 'racional';
}

function getSubtype(elem: unknown): string | null {
  if (!elem || typeof elem !== 'object') return null;
  const obj = elem as RuntimeElement;

  if (obj.kind === 'cpa' && obj.subtype) {
    return obj.subtype as string;
  }

  return null;
}

function getAttributes(elem: unknown): Record<string, string> {
  if (!elem || typeof elem !== 'object') return {};
  const obj = elem as RuntimeElement;

  if (obj.kind === 'cpa' && obj.attributes) {
    return obj.attributes as Record<string, string>;
  }

  return {};
}

// ============================================================================
// Sorting
// ============================================================================

/**
 * Sorts elements by quantity (stable sort).
 */
export function sortByQuantity(
  elements: unknown[],
  direction: 'asc' | 'desc'
): unknown[] {
  const indexed = elements.map((el, i) => ({ el, i }));
  indexed.sort((a, b) => {
    const qa = getQuantity(a.el);
    const qb = getQuantity(b.el);
    const cmp = qa - qb;
    if (cmp !== 0) return direction === 'asc' ? cmp : -cmp;
    return a.i - b.i; // stable sort by original index
  });
  return indexed.map((x) => x.el);
}

// ============================================================================
// Visual strip reconstruction
// ============================================================================

const MAX_VISUAL_UNITS = 48;

function flattenRuntimeElements(elements: unknown[]): unknown[] {
  const out: unknown[] = [];
  for (const el of elements) {
    if (!el || typeof el !== 'object') continue;
    const o = el as RuntimeElement;
    if (o.kind === 'arreglo' && Array.isArray(o.elements)) {
      out.push(...flattenRuntimeElements(o.elements as unknown[]));
    } else {
      out.push(el);
    }
  }
  return out;
}

/**
 * Reconstructs visualStrip from runtime elements.
 * Adapted from executeProgram.ts::buildVisualStrip
 */
export function buildVisualStripFromElements(elements: unknown[]): ResultVisualItem[] {
  const flat = flattenRuntimeElements(elements);
  const strip: ResultVisualItem[] = [];

  for (const elem of flat) {
    if (!elem || typeof elem !== 'object') continue;
    const o = elem as RuntimeElement;
    const rawAmt = getQuantity(elem);
    const n = Math.max(0, Math.min(24, Math.round(Number(rawAmt) || 0)));
    if (n === 0) continue;

    if (o.kind === 'cpa') {
      const type = o.type as string;
      const subtype = o.subtype as string;
      const attributes = (o.attributes as Record<string, string>) ?? {};
      const color = attributes.color ?? 'verde';
      const size = attributes.size ?? 'mediano';

      for (let i = 0; i < n; i++) {
        if (strip.length >= MAX_VISUAL_UNITS) return strip;

        switch (type) {
          case 'montessori':
            strip.push({ kind: 'montessori', color });
            break;
          case 'forma':
            strip.push({ kind: 'forma', subtype, size });
            break;
          case 'comida':
            strip.push({ kind: 'comida', subtype, color });
            break;
          case 'cap':
            strip.push({ kind: 'cap', color });
            break;
          case 'stick':
            strip.push({ kind: 'stick', color });
            break;
        }
      }
    }
  }

  return strip;
}

// ============================================================================
// Description generation
// ============================================================================

const PLURALS: Record<string, string> = {
  forma: 'formas',
  comida: 'comidas',
  montessori: 'montessoris',
  cap: 'tapas',
  stick: 'palitos',
  racional: 'racionales',
  cuadrado: 'cuadrados',
  circulo: 'círculos',
  triangulo: 'triángulos',
  rectangulo: 'rectángulos',
  rombo: 'rombos',
  estrella: 'estrellas',
  trapecio: 'trapecios',
  uva: 'uvas',
  pera: 'peras',
  manzana: 'manzanas',
  hamburguesa: 'hamburguesas',
  pasta: 'pastas',
};

function pluralize(word: string, count: number): string {
  if (count === 1) return word;
  return PLURALS[word] ?? word + 's';
}

interface SubtypeGroup {
  subtype: string;
  items: Array<{
    size?: string;
    color?: string;
    amount: number;
  }>;
  totalAmount: number;
}

interface TypeGroup {
  type: string;
  subtypes: SubtypeGroup[];
  totalAmount: number;
  rationalValue?: number;
}

interface CategoryGroup {
  category: 'abstracto' | 'pictorico' | 'concreto';
  types: TypeGroup[];
  totalAmount: number;
}

/**
 * Groups elements into categories/types/subtypes.
 * Adapted from executeProgram.ts::groupElements (partial)
 */
function groupElementsForDescription(elements: unknown[]): {
  categories: CategoryGroup[];
  totalAmount: number;
} {
  const categoryMap = new Map<string, CategoryGroup>();
  let totalAmount = 0;

  for (const elem of elements) {
    const amount = getQuantity(elem);
    totalAmount += amount;

    const category = getCategory(elem);
    const type = getType(elem);
    const subtype = getSubtype(elem);
    const attributes = getAttributes(elem);

    if (!categoryMap.has(category)) {
      categoryMap.set(category, {
        category,
        types: [],
        totalAmount: 0,
      });
    }
    const catGroup = categoryMap.get(category)!;
    catGroup.totalAmount += amount;

    let typeGroup = catGroup.types.find((t) => t.type === type);
    if (!typeGroup) {
      typeGroup = { type, subtypes: [], totalAmount: 0 };
      catGroup.types.push(typeGroup);
    }
    typeGroup.totalAmount += amount;

    if (type === 'racional') {
      typeGroup.rationalValue = (typeGroup.rationalValue ?? 0) + amount;
    }

    const effectiveSubtype = subtype ?? attributes.color ?? null;

    if (effectiveSubtype) {
      let subtypeGroup = typeGroup.subtypes.find((s) => s.subtype === effectiveSubtype);
      if (!subtypeGroup) {
        subtypeGroup = { subtype: effectiveSubtype, items: [], totalAmount: 0 };
        typeGroup.subtypes.push(subtypeGroup);
      }
      subtypeGroup.totalAmount += amount;

      subtypeGroup.items.push({
        size: attributes.size,
        color: attributes.color,
        amount,
      });
    }
  }

  return { categories: Array.from(categoryMap.values()), totalAmount };
}

function describeSubtype(sub: SubtypeGroup): string {
  const name = pluralize(sub.subtype, sub.totalAmount);

  if (sub.items.length === 1) {
    const item = sub.items[0];
    if (item.size) return `${sub.totalAmount} ${name} ${item.size}`;
    if (item.color) return `${sub.totalAmount} ${name} ${item.color}`;
    return `${sub.totalAmount} ${name}`;
  }

  if (sub.items.some((i) => i.size)) {
    const sizeDescs = sub.items.map((i) => `${i.amount} ${i.size}`);
    return `${sub.totalAmount} ${name} (${sizeDescs.join(', ')})`;
  }

  if (sub.items.some((i) => i.color)) {
    const colorDescs = sub.items.map((i) => `${i.amount} ${i.color}`);
    return `${sub.totalAmount} ${name} (${colorDescs.join(', ')})`;
  }

  return `${sub.totalAmount} ${name}`;
}

function describeType(type: TypeGroup): string {
  const typeName = pluralize(type.type, type.totalAmount);

  if (type.type === 'racional') {
    const val = type.rationalValue ?? type.totalAmount;
    if (type.totalAmount === 1) {
      return `el número ${val}`;
    }
    return `${type.totalAmount} ${typeName} (suma: ${val})`;
  }

  if (type.subtypes.length === 0) {
    return `${type.totalAmount} ${typeName}`;
  }

  if (type.subtypes.length === 1) {
    return describeSubtype(type.subtypes[0]);
  }

  const subDescs = type.subtypes.map((s) => describeSubtype(s));
  return `${type.totalAmount} ${typeName}: ${subDescs.join(', ')}`;
}

/**
 * Generates textual description from elements.
 * Adapted from executeProgram.ts::generateDescription
 */
export function generateDescriptionFromElements(elements: unknown[]): string {
  const { categories, totalAmount } = groupElementsForDescription(elements);

  if (categories.length === 0) return 'vacío';

  if (categories.length === 1) {
    const cat = categories[0];
    if (cat.types.length === 1) {
      return describeType(cat.types[0]);
    }
    const typeDescs = cat.types.map((t) => describeType(t));
    return `${totalAmount} objetos: ${typeDescs.join(', ')}`;
  }

  const parts = categories.map((cat) => {
    const typeDescs = cat.types.map((t) => describeType(t));
    return typeDescs.join(', ');
  });

  return `${totalAmount} objetos: ${parts.join('; ')}`;
}

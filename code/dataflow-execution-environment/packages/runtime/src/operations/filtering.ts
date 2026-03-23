import type { Shape, Car, Food, Animal, Person, Text } from "@dataflow/shared/types";

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

export function FILTER(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, value] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements;
  const filterValue = value.value;
  
  return { kind: "set", elements: elements.filter(item => deepEquals(item, filterValue)) };
}

export function FILTER_BY_SIZE(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, size] = inputs;
  const shapes = (set.value as { kind: string; elements: unknown[] }).elements as Array<{ size: string }>;
  const sizeValue = (size.value as { kind: string; value: string }).value;
  return { kind: "set", elements: shapes.filter(shape => shape.size === sizeValue) };
}

export function FILTER_BY_COLOR(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, color] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements as Array<{ color: string }>;
  const colorValue = (color.value as { kind: string; value: string }).value;
  return { kind: "set", elements: elements.filter(element => element.color === colorValue) };
}

export function FILTER_BY_TYPE(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, type] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements as Array<{ type: string }>;
  const typeValue = (type.value as { kind: string; value: string }).value;
  return { kind: "set", elements: elements.filter(element => element.type === typeValue) };
}

export function FILTER_BY_TASTE(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, taste] = inputs;
  const foods = (set.value as { kind: string; elements: unknown[] }).elements as Array<{ taste: string }>;
  const tasteValue = (taste.value as { kind: string; value: string }).value;
  return { kind: "set", elements: foods.filter(food => food.taste === tasteValue) };
}

export function FILTER_BY_AGE_GROUP(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, ageGroup] = inputs;
  const persons = (set.value as { kind: string; elements: unknown[] }).elements as Array<{ ageGroup: string }>;
  const ageGroupValue = (ageGroup.value as { kind: string; value: string }).value;
  return { kind: "set", elements: persons.filter(person => person.ageGroup === ageGroupValue) };
}

export function FILTER_BY_GENDER(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, gender] = inputs;
  const persons = (set.value as { kind: string; elements: unknown[] }).elements as Array<{ gender: string }>;
  const genderValue = (gender.value as { kind: string; value: string }).value;
  return { kind: "set", elements: persons.filter(person => person.gender === genderValue) };
}

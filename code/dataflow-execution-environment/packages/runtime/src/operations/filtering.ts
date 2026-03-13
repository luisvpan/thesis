import type { Shape, Car, Food, Animal, Person, Text } from "@dataflow/shared/types";

export function FILTER(inputs: Array<{ id: string; value: unknown }>): unknown {
  const [set, predicate] = inputs;
  const elements = (set.value as { kind: string; elements: unknown[] }).elements;
  const predicateFn = predicate.value as (item: unknown) => boolean;
  return { kind: "set", elements: elements.filter(predicateFn) };
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

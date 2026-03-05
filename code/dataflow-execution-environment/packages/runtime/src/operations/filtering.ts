import type { Shape, Car, Food, Animal, Person, Text } from "@dataflow/shared/types";

export function FILTER(inputs: Array<{ id: string; value: unknown }>): unknown[] {
  const [set, predicate] = inputs;
  const elements = set.value as unknown[];
  const predicateFn = predicate.value as (item: unknown) => boolean;
  return elements.filter(predicateFn);
}

export function FILTER_BY_SIZE(inputs: Array<{ id: string; value: unknown }>): Shape[] {
  const [set, size] = inputs;
  const shapes = set.value as Shape[];
  const sizeValue = (size.value as Text).value;
  return shapes.filter(shape => shape.size === sizeValue);
}

export function FILTER_BY_COLOR(inputs: Array<{ id: string; value: unknown }>): (Shape | Car | Food | Animal)[] {
  const [set, color] = inputs;
  const elements = set.value as (Shape | Car | Food | Animal)[];
  const colorValue = (color.value as Text).value;
  return elements.filter(element => element.color === colorValue);
}

export function FILTER_BY_TYPE(inputs: Array<{ id: string; value: unknown }>): (Shape | Animal)[] {
  const [set, type] = inputs;
  const elements = set.value as (Shape | Animal)[];
  const typeValue = (type.value as Text).value;
  return elements.filter(element => element.type === typeValue);
}

export function FILTER_BY_TASTE(inputs: Array<{ id: string; value: unknown }>): Food[] {
  const [set, taste] = inputs;
  const foods = set.value as Food[];
  const tasteValue = (taste.value as Text).value;
  return foods.filter(food => food.taste === tasteValue);
}

export function FILTER_BY_AGE_GROUP(inputs: Array<{ id: string; value: unknown }>): Person[] {
  const [set, ageGroup] = inputs;
  const persons = set.value as Person[];
  const ageGroupValue = (ageGroup.value as Text).value;
  return persons.filter(person => person.ageGroup === ageGroupValue);
}

export function FILTER_BY_GENDER(inputs: Array<{ id: string; value: unknown }>): Person[] {
  const [set, gender] = inputs;
  const persons = set.value as Person[];
  const genderValue = (gender.value as Text).value;
  return persons.filter(person => person.gender === genderValue);
}

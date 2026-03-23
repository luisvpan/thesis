import type { Shape, Car, Food, Animal, Person, Text, Boolean } from "@dataflow/shared/types";

export function COMPARE_BY_SIZE(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [s1, s2] = inputs;
  const size1 = (s1.value as Shape).size;
  const size2 = (s2.value as Shape).size;
  return { kind: "boolean" as const, value: size1 === size2 };
}

export function COMPARE_BY_COLOR(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [o1, o2] = inputs;
  const color1 = (o1.value as Shape | Car | Food | Animal).color;
  const color2 = (o2.value as Shape | Car | Food | Animal).color;
  return { kind: "boolean" as const, value: color1 === color2 };
}

export function COMPARE_BY_TYPE(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [o1, o2] = inputs;
  const type1 = (o1.value as Shape | Animal).type;
  const type2 = (o2.value as Shape | Animal).type;
  return { kind: "boolean" as const, value: type1 === type2 };
}

export function COMPARE_BY_TASTE(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [f1, f2] = inputs;
  const taste1 = (f1.value as Food).taste;
  const taste2 = (f2.value as Food).taste;
  return { kind: "boolean" as const, value: taste1 === taste2 };
}

export function COMPARE_BY_AGE_GROUP(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [p1, p2] = inputs;
  const ageGroup1 = (p1.value as Person).ageGroup;
  const ageGroup2 = (p2.value as Person).ageGroup;
  return { kind: "boolean" as const, value: ageGroup1 === ageGroup2 };
}

export function COMPARE_BY_GENDER(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [p1, p2] = inputs;
  const gender1 = (p1.value as Person).gender;
  const gender2 = (p2.value as Person).gender;
  return { kind: "boolean" as const, value: gender1 === gender2 };
}

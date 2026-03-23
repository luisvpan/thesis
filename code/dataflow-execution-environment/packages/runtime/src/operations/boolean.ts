import type { Boolean } from "@dataflow/shared/types";

export function AND(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [b1, b2] = inputs;
  const val1 = (b1.value as Boolean).value;
  const val2 = (b2.value as Boolean).value;
  return { kind: "boolean" as const, value: val1 && val2 };
}

export function OR(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [b1, b2] = inputs;
  const val1 = (b1.value as Boolean).value;
  const val2 = (b2.value as Boolean).value;
  return { kind: "boolean" as const, value: val1 || val2 };
}

export function NOT(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [b] = inputs;
  const val = (b.value as Boolean).value;
  return { kind: "boolean" as const, value: !val };
}

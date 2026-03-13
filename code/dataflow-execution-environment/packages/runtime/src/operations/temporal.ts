import { DataflowGraph } from "../graph/dataflow-graph.js";

export function NEXT(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [stream] = inputs;
  const streamValue = stream.value as { kind: "stream"; elementType: string; generator: Generator<unknown> };

  const gen = streamValue.generator;
  if (!gen) {
    throw new Error('Stream does not have a generator');
  }
  
  let result = gen.next();
  let count = 0;
  
  while (!result.done && count < time) {
    result = gen.next();
    count++;
  }
  
  return result.done ? undefined : result.value;
}

export function FIRST(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [stream] = inputs;
  const streamValue = stream.value as { kind: "stream"; elementType: string; firstValue: unknown };

  const value = streamValue.firstValue;
  if (typeof value === 'number') {
    return { kind: "natural", value };
  }
  return value;
}

export function FBY(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [initial, stream] = inputs;
  
  if (time === 0) {
    return initial.value;
  }
  
  const streamValue = stream.value as { kind: "stream"; elementType: string; generator: Generator<unknown> };

  const gen = streamValue.generator;
  if (!gen) {
    throw new Error('Stream does not have a generator');
  }
  
  let result = gen.next();
  let count = 0;
  
  while (!result.done && count < time - 1) {
    result = gen.next();
    count++;
  }
  
  return result.done ? initial.value : result.value;
}

export function ACCUMULATE(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [stream, initial, operation] = inputs;
  const streamValue = stream.value as { kind: "stream"; elementType: string; generator: Generator<unknown> };
  const initialValue = initial.value;
  const operationName = (operation.value as { kind: "text"; value: string }).value;
  
  if (!streamValue.generator) {
    throw new Error('Stream does not have a generator');
  }
  
  const gen = streamValue.generator;
  let accumulator = initialValue;
  let result = gen.next();
  let count = 0;
  
  while (!result.done && count < time) {
    accumulator = applyOperation(operationName, accumulator, result.value);
    result = gen.next();
    count++;
  }
  
  return accumulator;
}

function applyOperation(operationName: string, acc: unknown, value: unknown): unknown {
  switch (operationName) {
    case "ADD":
      return (acc as number) + (value as number);
    case "MULTIPLY":
      return (acc as number) * (value as number);
    default:
      throw new Error(`Unknown accumulation operation: ${operationName}`);
  }
}

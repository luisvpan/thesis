import type { DataflowGraph } from "../graph/dataflow-graph.js";
import type { DemandDrivenEvaluator } from "../evaluator/demand-driven-evaluator.js";

export async function NEXT(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator, currentNodeId: string): Promise<unknown> {
  const [streamInput] = inputs;

  const stream = await evaluator.evaluate(streamInput.id, time, graph) as { kind: "stream"; elementType: string; currentValue: unknown };
  
  if (!stream || typeof stream !== 'object' || !('currentValue' in stream)) {
    throw new Error('NEXT: Invalid stream value');
  }

  const currentValue = stream.currentValue;

  if (typeof currentValue !== 'number') {
    throw new Error(`NEXT: Expected numeric value, got ${typeof currentValue}`);
  }

  switch (stream.elementType) {
    case "natural":
      return { kind: "natural", value: currentValue };
    case "integer":
      return { kind: "integer", value: currentValue };
    case "decimal":
      return { kind: "decimal", value: currentValue };
    default:
      return currentValue;
  }
}

export async function FIRST(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator, currentNodeId: string): Promise<unknown> {
  const [stream] = inputs;

  const streamId = stream.id;
  const streamValue = await evaluator.evaluate(streamId, 0, graph) as { kind: "stream"; elementType: string; firstValue: unknown };

  if (typeof streamValue === 'object' && streamValue !== null && 'firstValue' in streamValue) {
    const { firstValue, elementType } = streamValue;

    switch (elementType) {
      case "natural":
        return { kind: "natural", value: firstValue as number };
      case "integer":
        return { kind: "integer", value: firstValue as number };
      case "decimal":
        return { kind: "decimal", value: firstValue as number };
      case "fraction":
        const frac = firstValue as { numerator: number; denominator: number };
        return { kind: "fraction", numerator: frac.numerator, denominator: frac.denominator };
      case "text":
        return { kind: "text", value: firstValue as string };
      case "boolean":
        return { kind: "boolean", value: firstValue as boolean };
      case "shape":
        const shape = firstValue as { kind: string; color: string; sides: number };
        return { kind: "shape", shape: shape.kind, color: shape.color, sides: shape.sides };
      case "car":
        const car = firstValue as { brand: string; color: string; speed: number };
        return { kind: "car", brand: car.brand, color: car.color, speed: car.speed };
      case "food":
        const food = firstValue as { kind: string; color: string; calories: number };
        return { kind: "food", type: food.kind, color: food.color, calories: food.calories };
      case "animal":
        const animal = firstValue as { species: string; color: string; legs: number };
        return { kind: "animal", species: animal.species, color: animal.color, legs: animal.legs };
      case "person":
        const person = firstValue as { name: string; age: number; color: string };
        return { kind: "person", name: person.name, age: person.age, hairColor: person.color };
      default:
        return firstValue;
    }
  }

  return streamValue;
}

export async function FBY(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator, currentNodeId: string): Promise<unknown> {
  const [initial, stream] = inputs;

  if (time === 0) {
    const initialId = initial.id;
    return await evaluator.evaluate(initialId, time, graph);
  }

  const streamId = stream.id;
  return await evaluator.evaluate(streamId, time - 1, graph);
}

export async function ACCUMULATE(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator, currentNodeId: string): Promise<unknown> {
  const [streamInput, operationInput, initialInput] = inputs;

  if (time === 0) {
    const initialId = initialInput.id;
    return await evaluator.evaluate(initialId, time, graph);
  }

  const stream = await evaluator.evaluate(streamInput.id, time, graph) as { kind: "stream"; elementType: string; currentValue: unknown };
  
  if (!stream || typeof stream !== 'object' || !('currentValue' in stream)) {
    throw new Error('ACCUMULATE: Invalid stream value');
  }

  const streamValue = stream.currentValue;
  const previousAccumulated = await evaluator.evaluate(currentNodeId, time - 1, graph);

  let operationName: string;
  if (typeof operationInput.value === 'string') {
    operationName = operationInput.value;
  } else if (typeof operationInput.value === 'object' && operationInput.value !== null && 'value' in operationInput.value) {
    operationName = (operationInput.value as { kind: "text"; value: string }).value;
  } else {
    throw new Error(`ACCUMULATE: Invalid operation input: ${JSON.stringify(operationInput.value)}`);
  }

  if (operationName === "ADD") {
    const kind = (previousAccumulated as any).kind || 'natural';
    const previousVal = (previousAccumulated as any).value ?? previousAccumulated;
    return { kind, value: (previousVal as number) + (streamValue as number) };
  }
  if (operationName === "MULTIPLY") {
    const kind = (previousAccumulated as any).kind || 'natural';
    const previousVal = (previousAccumulated as any).value ?? previousAccumulated;
    return { kind, value: (previousVal as number) * (streamValue as number) };
  }
  if (operationName === "SUBTRACT") {
    const kind = (previousAccumulated as any).kind || 'integer';
    const previousVal = (previousAccumulated as any).value ?? previousAccumulated;
    return { kind, value: (previousVal as number) - (streamValue as number) };
  }
  if (operationName === "DIVIDE") {
    if ((streamValue as number) === 0) {
      throw new Error("ACCUMULATE: Division by zero");
    }
    const kind = (previousAccumulated as any).kind || 'decimal';
    const previousVal = (previousAccumulated as any).value ?? previousAccumulated;
    return { kind, value: (previousVal as number) / (streamValue as number) };
  }

  throw new Error(`Unknown accumulation operation: ${operationName}`);
}

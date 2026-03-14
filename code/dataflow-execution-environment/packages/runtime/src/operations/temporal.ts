import type { DataflowGraph } from "../graph/dataflow-graph.js";
import type { DemandDrivenEvaluator } from "../evaluator/demand-driven-evaluator.js";

export function NEXT(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator): unknown {
  const [stream] = inputs;

  const streamId = stream.id;
  return evaluator.evaluate(streamId, time, graph);
}

export function FIRST(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator): unknown {
  const [stream] = inputs;

  const streamId = stream.id;
  const streamValue = evaluator.evaluate(streamId, 0, graph) as { kind: "stream"; elementType: string; firstValue: unknown };

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

export function FBY(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator): unknown {
  const [initial, stream] = inputs;

  if (time === 0) {
    const initialId = initial.id;
    return evaluator.evaluate(initialId, time, graph);
  }

  const streamId = stream.id;
  return evaluator.evaluate(streamId, time - 1, graph);
}

export function ACCUMULATE(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, evaluator: DemandDrivenEvaluator): unknown {
  const [stream, initial, operation] = inputs;

  if (time === 0) {
    const initialId = initial.id;
    return evaluator.evaluate(initialId, time, graph);
  }

  const streamId = stream.id;
  const streamValue = evaluator.evaluate(streamId, time - 1, graph);
  const previousAccumulated = evaluator.evaluate(streamId, time - 1, graph);

  const operationName = (operation.value as { kind: "text"; value: string }).value;

  if (operationName === "ADD") {
    return (previousAccumulated as number) + (streamValue as number);
  }
  if (operationName === "MULTIPLY") {
    return (previousAccumulated as number) * (streamValue as number);
  }
  if (operationName === "SUBTRACT") {
    return (previousAccumulated as number) - (streamValue as number);
  }
  if (operationName === "DIVIDE") {
    if ((streamValue as number) === 0) {
      throw new Error("ACCUMULATE: Division by zero");
    }
    return (previousAccumulated as number) / (streamValue as number);
  }

  throw new Error(`Unknown accumulation operation: ${operationName}`);
}

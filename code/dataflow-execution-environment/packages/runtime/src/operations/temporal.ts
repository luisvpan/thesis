import { DataflowGraph } from "../graph/dataflow-graph.js";

export function NEXT(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [stream] = inputs;
  
  const streamId = stream.id;
  return graph.evaluate(streamId, time, graph);
}

export function FIRST(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [stream] = inputs;
  
  const streamId = stream.id;
  const streamValue = graph.evaluate(streamId, 0, graph) as { kind: "stream"; elementType: string; firstValue: unknown };
  
  if (typeof streamValue === 'object' && streamValue !== null && 'firstValue' in streamValue) {
    return streamValue.firstValue;
  }
  
  return streamValue;
}

export function FBY(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [initial, stream] = inputs;
  
  if (time === 0) {
    return initial.value;
  }
  
  const streamId = stream.id;
  return graph.evaluate(streamId, time - 1, graph);
}

export function ACCUMULATE(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [stream, initial, operation] = inputs;
  
  if (time === 0) {
    return initial.value;
  }
  
  const streamId = stream.id;
  const streamValue = graph.evaluate(streamId, time - 1, graph);
  const previousAccumulated = graph.evaluate(streamId, time - 1, graph);
  
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

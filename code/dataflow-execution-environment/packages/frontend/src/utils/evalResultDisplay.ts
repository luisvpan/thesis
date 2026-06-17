import type { ProgramOutputFlowNodeData } from "@/components/dataflow";
import type { ResultValue } from "@/services/executeProgram";

export type FlowResultDisplayData = Pick<
  ProgramOutputFlowNodeData,
  | "value"
  | "description"
  | "visualStrip"
  | "originalElements"
  | "isSingleCpaObject"
  | "singleCpaObjectMeta"
  | "numerator"
  | "denominator"
  | "numberArrayValues"
  | "booleanValue"
>;

export function resultValueToDisplayData(
  resultValue: ResultValue
): FlowResultDisplayData {
  if (resultValue.kind === "boolean") {
    return {
      value: undefined,
      description: resultValue.value ? "verdadero" : "falso",
      booleanValue: resultValue.value,
      visualStrip: undefined,
    };
  }
  if (resultValue.kind === "number") {
    return {
      value: resultValue.value,
      description: undefined,
      visualStrip: undefined,
      numerator: resultValue.numerator,
      denominator: resultValue.denominator,
    };
  }
  if (resultValue.kind === "numberArray") {
    return {
      value: undefined,
      description: undefined,
      visualStrip: undefined,
      numberArrayValues: resultValue.values,
    };
  }
  if (resultValue.kind !== "semantic") {
    return { value: undefined, description: undefined, visualStrip: undefined };
  }
  return {
    value: resultValue.result.totalAmount,
    description: resultValue.result.description,
    visualStrip: resultValue.result.visualStrip,
    originalElements: resultValue.result.originalElements,
    isSingleCpaObject: resultValue.isSingleCpaObject,
    singleCpaObjectMeta: resultValue.singleCpaObjectMeta,
  };
}

export function hasFlowResultDisplay(data: FlowResultDisplayData): boolean {
  return (
    data.value !== undefined ||
    data.booleanValue !== undefined ||
    Boolean(data.description) ||
    (data.visualStrip != null && data.visualStrip.length > 0) ||
    (data.numberArrayValues != null && data.numberArrayValues.length > 0)
  );
}

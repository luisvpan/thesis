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
>;

export function resultValueToDisplayData(
  resultValue: ResultValue
): FlowResultDisplayData {
  if (resultValue.kind === "number") {
    return {
      value: resultValue.value,
      description: undefined,
      visualStrip: undefined,
      numerator: resultValue.numerator,
      denominator: resultValue.denominator,
    };
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
    Boolean(data.description) ||
    (data.visualStrip != null && data.visualStrip.length > 0)
  );
}

import { describe, expect, test } from "bun:test";
import type { DataflowNode } from "@/contexts/node/types";
import { FLOW_CARD_SIZE } from "./arrayZoneGeometry";
import {
  applyNumberTouchMerge,
  getNumberTouchGroups,
  isNumberMergeHead,
  isNumberMergeTail,
  resolveNumberSourceId,
  shouldEmitNumberSource,
} from "./numberTouchMerge";
import type { SourceFlowNodeData } from "@/components/dataflow";

function numberNode(
  id: string,
  digit: number,
  x: number,
  y = 0
): DataflowNode {
  return {
    id,
    type: "source",
    position: { x, y },
    data: { variant: "number", value: digit, digitValue: digit },
  };
}

describe("getNumberTouchGroups", () => {
  test("two touching number cards merge left-to-right as 12", () => {
    const nodes = [
      numberNode("a", 1, 0),
      numberNode("b", 2, FLOW_CARD_SIZE - 20),
    ];
    const groups = getNumberTouchGroups(nodes);
    expect(groups).toHaveLength(1);
    expect(groups[0].mergedValue).toBe(12);
    expect(groups[0].primaryId).toBe("a");
    expect(groups[0].tailId).toBe("b");
    expect(groups[0].memberIds).toEqual(["a", "b"]);
  });

  test("three touching cards concatenate to 123", () => {
    const nodes = [
      numberNode("a", 1, 0),
      numberNode("b", 2, FLOW_CARD_SIZE - 20),
      numberNode("c", 3, 2 * (FLOW_CARD_SIZE - 20)),
    ];
    const groups = getNumberTouchGroups(nodes);
    expect(groups).toHaveLength(1);
    expect(groups[0].mergedValue).toBe(123);
  });

  test("separated numbers stay in distinct groups", () => {
    const nodes = [
      numberNode("a", 1, 0),
      numberNode("b", 2, FLOW_CARD_SIZE * 3),
    ];
    const groups = getNumberTouchGroups(nodes);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.mergedValue).sort((x, y) => x - y)).toEqual([
      1, 2,
    ]);
  });
});

describe("applyNumberTouchMerge", () => {
  test("sets merged value on all members", () => {
    const nodes = [
      numberNode("a", 1, 0),
      numberNode("b", 2, FLOW_CARD_SIZE - 20),
    ];
    const merged = applyNumberTouchMerge(nodes);
    expect((merged[0].data as { value: number }).value).toBe(12);
    expect((merged[1].data as { value: number }).value).toBe(12);
    expect((merged[0].data as { digitValue: number }).digitValue).toBe(1);
    expect((merged[1].data as { digitValue: number }).digitValue).toBe(2);
  });

  test("marks head and tail ids for chrome (header / handle)", () => {
    const nodes = applyNumberTouchMerge([
      numberNode("a", 1, 0),
      numberNode("b", 2, FLOW_CARD_SIZE - 20),
      numberNode("c", 3, 2 * (FLOW_CARD_SIZE - 20)),
    ]);
    const da = nodes[0].data as SourceFlowNodeData;
    const db = nodes[1].data as SourceFlowNodeData;
    const dc = nodes[2].data as SourceFlowNodeData;
    expect(da.variant === "number" && da.numberMergePrimaryId).toBe("a");
    expect(da.variant === "number" && da.numberMergeTailId).toBe("c");
    expect(isNumberMergeHead("a", da)).toBe(true);
    expect(isNumberMergeHead("b", db)).toBe(false);
    expect(isNumberMergeTail("c", dc)).toBe(true);
    expect(isNumberMergeTail("a", da)).toBe(false);
  });
});

describe("program emission helpers", () => {
  test("only primary id should emit source", () => {
    const nodes = [
      numberNode("a", 1, 0),
      numberNode("b", 2, FLOW_CARD_SIZE - 20),
    ];
    expect(shouldEmitNumberSource("a", nodes)).toBe(true);
    expect(shouldEmitNumberSource("b", nodes)).toBe(false);
    expect(resolveNumberSourceId("b", nodes)).toBe("a");
  });
});

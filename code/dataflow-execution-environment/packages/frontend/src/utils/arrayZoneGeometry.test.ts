import { describe, expect, test } from "bun:test";
import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "../contexts/node/types";
import { flowToProgram } from "./flowToProgram";
import {
  ARRAY_CLOSE_ZONE_IN_TOP_FRAC,
  FLOW_CARD_SIZE,
  FLOW_HANDLE_SIZE,
  computeNodeIdsInsideActiveArrayZones,
  getArrayCloseZoneInCenter,
  getArrayOpenZoneOutCenter,
  getArrayZoneBounds,
  getFlowCardBounds,
  getFlowCardCenter,
  getOrderedArrayZoneMembers,
  doAxisAlignedBoundsOverlap,
  isPointInBoundsInclusive,
  shouldIncludeNodeInArrayZone,
} from "./arrayZoneGeometry";

describe("arrayZoneGeometry", () => {
  test("getArrayOpenZoneOutCenter uses FLOW_CARD_SIZE and 100px offset", () => {
    const c = getArrayOpenZoneOutCenter({ position: { x: 10, y: 20 } });
    expect(c.x).toBe(10 + FLOW_CARD_SIZE + 100);
    expect(c.y).toBe(20 + FLOW_CARD_SIZE / 2);
  });

  test("getArrayCloseZoneInCenter uses zone-in fraction and left handle offset", () => {
    const c = getArrayCloseZoneInCenter({ position: { x: 400, y: 200 } });
    expect(c.x).toBe(400 - 100 + FLOW_HANDLE_SIZE / 2);
    expect(c.y).toBe(200 + ARRAY_CLOSE_ZONE_IN_TOP_FRAC * FLOW_CARD_SIZE);
  });

  test("getArrayZoneBounds unions both handle AABBs (80×80)", () => {
    const open = { position: { x: 0, y: 0 } };
    const close = { position: { x: 400, y: 200 } };
    const b = getArrayZoneBounds(open, close);
    const p0 = getArrayOpenZoneOutCenter(open);
    const p1 = getArrayCloseZoneInCenter(close);
    const half = FLOW_HANDLE_SIZE / 2;
    expect(b.left).toBe(Math.min(p0.x, p1.x) - half);
    expect(b.right).toBe(Math.max(p0.x, p1.x) + half);
    expect(b.top).toBe(Math.min(p0.y, p1.y) - half);
    expect(b.bottom).toBe(Math.max(p0.y, p1.y) + half);
  });

  test("shouldIncludeNodeInArrayZone excludes pair ends and programOutput", () => {
    const open = { position: { x: 0, y: 0 } };
    const close = { position: { x: 400, y: 200 } };
    const bounds = getArrayZoneBounds(open, close);

    const openNode = {
      id: "open1",
      type: "arrayOpen",
      position: { x: 0, y: 0 },
      data: {},
    } as DataflowNode;

    const closeNode = {
      id: "close1",
      type: "arrayClose",
      position: { x: 400, y: 200 },
      data: {},
    } as DataflowNode;

    const outNode = {
      id: "out1",
      type: "programOutput",
      position: { x: 300, y: 150 },
      data: {},
    } as DataflowNode;

    const insideCenter = getFlowCardCenter({ position: { x: 216, y: 76 } });
    expect(isPointInBoundsInclusive(insideCenter, bounds)).toBe(true);

    expect(
      shouldIncludeNodeInArrayZone(openNode, "open1", "close1", bounds)
    ).toBe(false);
    expect(
      shouldIncludeNodeInArrayZone(closeNode, "open1", "close1", bounds)
    ).toBe(false);
    expect(
      shouldIncludeNodeInArrayZone(outNode, "open1", "close1", bounds)
    ).toBe(false);
  });

  test("shouldIncludeNodeInArrayZone when card touches zone edge but center is outside", () => {
    const open = { position: { x: 0, y: 0 } };
    const close = { position: { x: 400, y: 200 } };
    const bounds = getArrayZoneBounds(open, close);

    const touchingEdge: DataflowNode = {
      id: "s_touch",
      type: "source",
      position: { x: 100, y: 150 },
      data: { variant: "number", value: 1 },
    } as DataflowNode;

    const cardBounds = getFlowCardBounds(touchingEdge);
    expect(doAxisAlignedBoundsOverlap(cardBounds, bounds)).toBe(true);
    expect(
      isPointInBoundsInclusive(getFlowCardCenter(touchingEdge), bounds)
    ).toBe(false);
    expect(
      shouldIncludeNodeInArrayZone(touchingEdge, "open1", "close1", bounds)
    ).toBe(true);
  });

  test("computeNodeIdsInsideActiveArrayZones lists sources inside any connected zone", () => {
    const open: DataflowNode = {
      id: "o1",
      type: "arrayOpen",
      position: { x: 0, y: 0 },
      data: {},
    } as DataflowNode;
    const close: DataflowNode = {
      id: "c1",
      type: "arrayClose",
      position: { x: 400, y: 200 },
      data: {},
    } as DataflowNode;
    const inside: DataflowNode = {
      id: "s_in",
      type: "source",
      position: { x: 216, y: 76 },
      data: { variant: "number", value: 1 },
    } as DataflowNode;
    const outside: DataflowNode = {
      id: "s_out",
      type: "source",
      position: { x: 0, y: 0 },
      data: { variant: "number", value: 2 },
    } as DataflowNode;
    const edge: Edge = {
      id: "e",
      source: "o1",
      target: "c1",
      sourceHandle: "zone-out",
      targetHandle: "zone-in",
    };
    const set = computeNodeIdsInsideActiveArrayZones(
      [open, close, inside, outside],
      [edge]
    );
    expect(set.has("s_in")).toBe(true);
    expect(set.has("s_out")).toBe(false);
    expect(set.has("o1")).toBe(false);
    expect(set.has("c1")).toBe(false);
  });
});

describe("flowToProgram array zone", () => {
  test("array literal lists only source nodes whose card overlaps handler AABB", () => {
    const open: DataflowNode = {
      id: "open1",
      type: "arrayOpen",
      position: { x: 0, y: 0 },
      data: {},
    };
    const close: DataflowNode = {
      id: "close1",
      type: "arrayClose",
      position: { x: 400, y: 200 },
      data: {},
    };
    const inside: DataflowNode = {
      id: "src_in",
      type: "source",
      position: { x: 216, y: 76 },
      data: { variant: "number", value: 1 },
    };
    const outside: DataflowNode = {
      id: "src_out",
      type: "source",
      position: { x: 0, y: 0 },
      data: { variant: "number", value: 2 },
    };

    const edge: Edge = {
      id: "e1",
      source: "open1",
      target: "close1",
      sourceHandle: "zone-out",
      targetHandle: "zone-in",
    };

    const program = flowToProgram([open, close, inside, outside], [edge]);
    const arrayStmt = program.statements.find(
      (s) =>
        s.type === "SourceStatement" &&
        s.identifier === "close1" &&
        s.value.type === "ArrayLiteral"
    );
    expect(arrayStmt).toBeDefined();
    if (!arrayStmt || arrayStmt.type !== "SourceStatement") return;
    if (arrayStmt.value.type !== "ArrayLiteral") return;
    const names = arrayStmt.value.elements.map((e) =>
      e.type === "Identifier" ? e.name : ""
    );
    expect(names).toEqual(["src_in"]);
    expect(names).not.toContain("src_out");
  });

  test("getOrderedArrayZoneMembers matches flowToProgram array order", () => {
    // Position nodes so they fall inside the array zone
    // Zone bounds: openZoneOut to closeZoneIn
    // openZoneOut: open.x + FLOW_CARD_SIZE + 100 = 0 + 208 + 100 = 308
    // closeZoneIn: close.x - 100 + 40 = 800 - 100 + 40 = 740
    // Zone y: ~104 (open center) to ~162 (close zone-in)
    const open: DataflowNode = {
      id: "open1",
      type: "arrayOpen",
      position: { x: 0, y: 0 },
      data: {},
    };
    const close: DataflowNode = {
      id: "close1",
      type: "arrayClose",
      position: { x: 800, y: 50 },
      data: {},
    };
    // Place nodes so their centers fall inside the zone
    // Center = position + FLOW_CARD_SIZE/2 = position + 104
    const first: DataflowNode = {
      id: "a",
      type: "source",
      position: { x: 350, y: 0 },  // center at 454, inside [308, 740]
      data: { variant: "number", value: 1 },
    };
    const second: DataflowNode = {
      id: "b",
      type: "source",
      position: { x: 500, y: 0 },  // center at 604, inside [308, 740]
      data: { variant: "number", value: 2 },
    };
    const edge: Edge = {
      id: "e1",
      source: "open1",
      target: "close1",
      sourceHandle: "zone-out",
      targetHandle: "zone-in",
    };
    const members = getOrderedArrayZoneMembers("close1", [open, close, first, second], [edge]);
    expect(members.map((n) => n.id)).toEqual(["a", "b"]);
  });
});

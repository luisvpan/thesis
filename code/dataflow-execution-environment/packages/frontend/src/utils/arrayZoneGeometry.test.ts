import { describe, expect, test } from "bun:test";
import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "../contexts/node/types";
import { flowToProgram } from "./flowToProgram";
import {
  ARRAY_NODE_HEIGHT,
  ARRAY_NODE_WIDTH,
  ARRAY_ZONE_HANDLE_OFFSET_X,
} from "../components/dataflow/arrayNodeLayout";
import {
  ARRAY_CLOSE_ZONE_IN_TOP_FRAC,
  FLOW_HANDLE_SIZE,
  computeNodeIdsInsideActiveArrayZones,
  getArrayCloseZoneInCenter,
  getArrayOpenZoneOutCenter,
  getArrayZoneBounds,
  getFlowCardCenter,
  isPointInBoundsInclusive,
  shouldIncludeNodeInArrayZone,
} from "./arrayZoneGeometry";

describe("arrayZoneGeometry", () => {
  test("getArrayOpenZoneOutCenter uses array node size and zone-out handle offset", () => {
    const c = getArrayOpenZoneOutCenter({ position: { x: 10, y: 20 } });
    expect(c.x).toBe(10 + ARRAY_NODE_WIDTH + ARRAY_ZONE_HANDLE_OFFSET_X);
    expect(c.y).toBe(20 + ARRAY_NODE_HEIGHT / 2);
  });

  test("getArrayCloseZoneInCenter uses zone-in fraction and left handle offset", () => {
    const c = getArrayCloseZoneInCenter({ position: { x: 400, y: 200 } });
    expect(c.x).toBe(400 - ARRAY_ZONE_HANDLE_OFFSET_X + FLOW_HANDLE_SIZE / 2);
    expect(c.y).toBe(
      200 +
        ARRAY_CLOSE_ZONE_IN_TOP_FRAC * ARRAY_NODE_HEIGHT +
        FLOW_HANDLE_SIZE / 2
    );
  });

  test("getArrayZoneBounds is axis-aligned min/max of both handle centers", () => {
    const open = { position: { x: 0, y: 0 } };
    const close = { position: { x: 400, y: 200 } };
    const b = getArrayZoneBounds(open, close);
    const p0 = getArrayOpenZoneOutCenter(open);
    const p1 = getArrayCloseZoneInCenter(close);
    expect(b.left).toBe(Math.min(p0.x, p1.x));
    expect(b.right).toBe(Math.max(p0.x, p1.x));
    expect(b.top).toBe(Math.min(p0.y, p1.y));
    expect(b.bottom).toBe(Math.max(p0.y, p1.y));
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
  test("array literal lists only source nodes whose card center lies in handler AABB", () => {
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
});

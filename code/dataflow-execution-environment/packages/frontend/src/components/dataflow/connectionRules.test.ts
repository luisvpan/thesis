import { describe, expect, test } from "bun:test";
import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/node/types";
import {
  canConnectPorts,
  canConnectStructurally,
  isArrayClosePairedWithOpen,
  isPortOccupied,
  wouldPortsConnect,
} from "./connectionRules";

const noopKind = () => undefined;

function nodes(
  ...specs: Array<{ id: string; type: DataflowNode["type"]; data?: Record<string, unknown> }>
): DataflowNode[] {
  return specs.map((s) => ({
    id: s.id,
    type: s.type,
    position: { x: 0, y: 0 },
    data: s.data ?? {},
  })) as DataflowNode[];
}

describe("connectionRules", () => {
  test("isPortOccupied detects source and target edges", () => {
    const edges: Edge[] = [
      { id: "e1", source: "s1", target: "o1", sourceHandle: "out", targetHandle: "a" },
    ];
    expect(
      isPortOccupied(edges, { nodeId: "s1", handleId: "out", handleType: "source" })
    ).toBe(true);
    expect(
      isPortOccupied(edges, { nodeId: "o1", handleId: "a", handleType: "target" })
    ).toBe(true);
    expect(
      isPortOccupied(edges, { nodeId: "o1", handleId: "b", handleType: "target" })
    ).toBe(false);
  });

  test("(a) source out only connects to operator a|b", () => {
    const ns = nodes(
      { id: "s", type: "source" },
      { id: "o", type: "operator" },
      { id: "p", type: "programOutput" }
    );
    const ctx = { nodes: ns, edges: [] as Edge[] };

    expect(
      canConnectStructurally(
        { nodeId: "s", handleId: "out", handleType: "source" },
        { nodeId: "o", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);

    expect(
      canConnectStructurally(
        { nodeId: "s", handleId: "out", handleType: "source" },
        { nodeId: "p", handleId: "in", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);
  });

  test("arrayOpen zone-out only connects to arrayClose zone-in", () => {
    const ns = nodes(
      { id: "open", type: "arrayOpen" },
      { id: "close", type: "arrayClose" },
      { id: "o", type: "operator" }
    );
    const ctx = { nodes: ns, edges: [] as Edge[] };

    expect(
      canConnectStructurally(
        { nodeId: "open", handleId: "zone-out", handleType: "source" },
        { nodeId: "close", handleId: "zone-in", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);

    expect(
      canConnectStructurally(
        { nodeId: "open", handleId: "zone-out", handleType: "source" },
        { nodeId: "o", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);
  });

  test("arrayClose out to operator requires open pair", () => {
    const ns = nodes(
      { id: "open", type: "arrayOpen" },
      { id: "close", type: "arrayClose" },
      { id: "o", type: "operator" }
    );
    const noPair: Edge[] = [];
    const withPair: Edge[] = [
      {
        id: "z",
        source: "open",
        target: "close",
        sourceHandle: "zone-out",
        targetHandle: "zone-in",
      },
    ];

    const src = { nodeId: "close", handleId: "out", handleType: "source" as const };
    const tgt = { nodeId: "o", handleId: "a", handleType: "target" as const };

    expect(canConnectStructurally(src, tgt, { nodes: ns, edges: noPair }).ok).toBe(false);
    expect(canConnectStructurally(src, tgt, { nodes: ns, edges: withPair }).ok).toBe(true);
    expect(isArrayClosePairedWithOpen("close", withPair, ns)).toBe(true);
  });

  test("arrayClose out to programOutput does not require open pair", () => {
    const ns = nodes({ id: "close", type: "arrayClose" }, { id: "p", type: "programOutput" });
    expect(
      canConnectStructurally(
        { nodeId: "close", handleId: "out", handleType: "source" },
        { nodeId: "p", handleId: "in", handleType: "target" },
        { nodes: ns, edges: [] }
      ).ok
    ).toBe(true);
  });

  test("programOutput out only connects to operator inputs", () => {
    const ns = nodes({ id: "p", type: "programOutput" }, { id: "o", type: "operator" });
    const ctx = { nodes: ns, edges: [] as Edge[] };

    expect(
      canConnectStructurally(
        { nodeId: "p", handleId: "out", handleType: "source" },
        { nodeId: "o", handleId: "b", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);

    expect(
      canConnectStructurally(
        { nodeId: "p", handleId: "out", handleType: "source" },
        { nodeId: "p", handleId: "in", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);
  });

  test("(e) occupied ports block canConnectPorts", () => {
    const ns = nodes({ id: "s", type: "source" }, { id: "o", type: "operator" });
    const edges: Edge[] = [
      { id: "e1", source: "s", target: "o", sourceHandle: "out", targetHandle: "a" },
    ];
    const ctx = { nodes: ns, edges, getPortKindInfo: noopKind };

    expect(
      canConnectPorts(
        { nodeId: "s", handleId: "out", handleType: "source" },
        { nodeId: "o", handleId: "b", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);
  });

  test("order operator accepts only arrayClose out", () => {
    const ns = nodes(
      { id: "open", type: "arrayOpen" },
      { id: "close", type: "arrayClose" },
      { id: "s", type: "source" },
      { id: "ord", type: "operator", data: { operator: "orden-menor-mayor" } }
    );
    const zonePair: Edge[] = [
      {
        id: "z",
        source: "open",
        target: "close",
        sourceHandle: "zone-out",
        targetHandle: "zone-in",
      },
    ];
    const ctx = { nodes: ns, edges: zonePair };

    expect(
      canConnectStructurally(
        { nodeId: "close", handleId: "out", handleType: "source" },
        { nodeId: "ord", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);

    expect(
      canConnectStructurally(
        { nodeId: "s", handleId: "out", handleType: "source" },
        { nodeId: "ord", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);

    expect(
      canConnectStructurally(
        { nodeId: "close", handleId: "out", handleType: "source" },
        { nodeId: "ord", handleId: "b", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);
  });

  test("wouldPortsConnect mirrors canConnectPorts for valid pair", () => {
    const ns = nodes({ id: "s", type: "source" }, { id: "o", type: "operator" });
    const ctx = { nodes: ns, edges: [] as Edge[], getPortKindInfo: noopKind };
    const a = { nodeId: "s", handleId: "out", handleType: "source" as const };
    const b = { nodeId: "o", handleId: "a", handleType: "target" as const };
    expect(wouldPortsConnect(a, b, ctx)).toBe(true);
  });

  test("filter operator accepts arrayClose group on input a", () => {
    const ns = nodes(
      { id: "open", type: "arrayOpen" },
      { id: "close", type: "arrayClose" },
      { id: "flt", type: "operator", data: { operator: "filtrar-general" } }
    );
    const zonePair: Edge[] = [
      {
        id: "z",
        source: "open",
        target: "close",
        sourceHandle: "zone-out",
        targetHandle: "zone-in",
      },
    ];
    const ctx = {
      nodes: ns,
      edges: zonePair,
      getPortKindInfo: (nodeId: string, handleId: string) => {
        if (nodeId === "close" && handleId === "out") return { produces: "group" as const };
        if (nodeId === "flt" && handleId === "a") return { accepts: ["group", "cpa"] as const };
        return undefined;
      },
    };

    expect(
      canConnectPorts(
        { nodeId: "close", handleId: "out", handleType: "source" },
        { nodeId: "flt", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);
  });

  test("addition accepts arrayClose group on inputs a and b", () => {
    const ns = nodes(
      { id: "open", type: "arrayOpen" },
      { id: "close", type: "arrayClose" },
      { id: "add", type: "operator", data: { operator: "adicion" } }
    );
    const zonePair: Edge[] = [
      {
        id: "z",
        source: "open",
        target: "close",
        sourceHandle: "zone-out",
        targetHandle: "zone-in",
      },
    ];
    const ctx = {
      nodes: ns,
      edges: zonePair,
      getPortKindInfo: (nodeId: string, handleId: string) => {
        if (nodeId === "close" && handleId === "out") return { produces: "group" as const };
        if (nodeId === "add" && (handleId === "a" || handleId === "b")) {
          return { accepts: ["group", "cpa", "rational"] as const };
        }
        return undefined;
      },
    };

    expect(
      canConnectPorts(
        { nodeId: "close", handleId: "out", handleType: "source" },
        { nodeId: "add", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);

    expect(
      canConnectPorts(
        { nodeId: "close", handleId: "out", handleType: "source" },
        { nodeId: "add", handleId: "b", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);
  });

  test("addition rejects criteria keyword on inputs", () => {
    const ns = nodes(
      { id: "red", type: "source", data: { variant: "criteria", yoloClass: "red" } },
      { id: "add", type: "operator", data: { operator: "adicion" } }
    );
    const ctx = {
      nodes: ns,
      edges: [] as Edge[],
      getPortKindInfo: (nodeId: string, handleId: string) => {
        if (nodeId === "red" && handleId === "out") return { produces: "keyword" as const };
        if (nodeId === "add" && (handleId === "a" || handleId === "b")) {
          return { accepts: ["group", "cpa", "rational"] as const };
        }
        return undefined;
      },
    };

    expect(
      canConnectPorts(
        { nodeId: "red", handleId: "out", handleType: "source" },
        { nodeId: "add", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);
  });

  test("addition accepts shape CPA but not standalone shape criteria", () => {
    const ns = nodes(
      { id: "tri", type: "source", data: { variant: "shape", yoloClass: "lg_triangle" } },
      { id: "crit", type: "source", data: { variant: "criteria", yoloClass: "triangle" } },
      { id: "add", type: "operator", data: { operator: "adicion" } }
    );
    const ctx = {
      nodes: ns,
      edges: [] as Edge[],
      getPortKindInfo: (nodeId: string, handleId: string) => {
        if (nodeId === "tri" && handleId === "out") return { produces: "cpa" as const };
        if (nodeId === "crit" && handleId === "out") return { produces: "keyword" as const };
        if (nodeId === "add" && (handleId === "a" || handleId === "b")) {
          return { accepts: ["group", "cpa", "rational"] as const };
        }
        return undefined;
      },
    };

    expect(
      canConnectPorts(
        { nodeId: "tri", handleId: "out", handleType: "source" },
        { nodeId: "add", handleId: "a", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);

    expect(
      canConnectPorts(
        { nodeId: "crit", handleId: "out", handleType: "source" },
        { nodeId: "add", handleId: "b", handleType: "target" },
        ctx
      ).ok
    ).toBe(false);
  });

  test("filter operator accepts criteria keyword on input b", () => {
    const ns = nodes(
      { id: "crit", type: "source", data: { variant: "criteria" } },
      { id: "flt", type: "operator", data: { operator: "filtrar-general" } }
    );
    const ctx = {
      nodes: ns,
      edges: [] as Edge[],
      getPortKindInfo: (nodeId: string, handleId: string) => {
        if (nodeId === "crit" && handleId === "out") return { produces: "keyword" as const };
        if (nodeId === "flt" && handleId === "b") return { accepts: ["keyword"] as const };
        return undefined;
      },
    };

    expect(
      canConnectPorts(
        { nodeId: "crit", handleId: "out", handleType: "source" },
        { nodeId: "flt", handleId: "b", handleType: "target" },
        ctx
      ).ok
    ).toBe(true);
  });
});

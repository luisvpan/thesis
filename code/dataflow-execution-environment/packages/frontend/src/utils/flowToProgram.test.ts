import { describe, expect, test } from "bun:test";
import { Interpreter } from "@dataflow/interpreter";
import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/node/types";
import { flowToProgram, resolveFlowSourceId } from "./flowToProgram";

function numericValue(value: {
  kind: string;
  value?: { valueOf(): number | bigint };
  quantity?: { valueOf(): number | bigint };
}): number {
  if (value.kind === "racional" && value.value) return Number(value.value.valueOf());
  if (value.kind === "cpa" && value.quantity) return Number(value.quantity.valueOf());
  throw new Error(`unexpected runtime kind: ${value.kind}`);
}

function numberSource(id: string, value: number, x: number): DataflowNode {
  return {
    id,
    type: "source",
    position: { x, y: 0 },
    data: { variant: "number", value },
  };
}

function operator(
  id: string,
  op: "adicion" | "sustraccion" | "orden-menor-mayor" | "orden-mayor-menor" = "adicion"
): DataflowNode {
  return {
    id,
    type: "operator",
    position: { x: 0, y: 0 },
    data: { operator: op },
  };
}

function programOutput(id: string): DataflowNode {
  return {
    id,
    type: "programOutput",
    position: { x: 0, y: 0 },
    data: {},
  };
}

describe("resolveFlowSourceId", () => {
  test("programOutput maps to output_ sink identifier", () => {
    const nodes = [programOutput("out1")];
    expect(resolveFlowSourceId("out1", nodes)).toBe("output_out1");
  });
});

describe("flowToProgram programOutput chain", () => {
  test("(5 - 4) -> output1 -> (+ 4) -> output2 executes without undefined reference", async () => {
    const n5 = numberSource("n5", 5, 0);
    const n4a = numberSource("n4a", 4, 300);
    const n4b = numberSource("n4b", 4, 600);
    const sub = operator("sub", "sustraccion");
    const out1 = programOutput("out1");
    const add = operator("add", "adicion");
    const out2 = programOutput("out2");

    const nodes = [n5, n4a, n4b, sub, out1, add, out2];
    const edges: Edge[] = [
      { id: "e1", source: "n5", target: "sub", sourceHandle: "out", targetHandle: "a" },
      { id: "e2", source: "n4a", target: "sub", sourceHandle: "out", targetHandle: "b" },
      { id: "e3", source: "sub", target: "out1", sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "out1", target: "add", sourceHandle: "out", targetHandle: "a" },
      { id: "e5", source: "n4b", target: "add", sourceHandle: "out", targetHandle: "b" },
      { id: "e6", source: "add", target: "out2", sourceHandle: "out", targetHandle: "in" },
    ];

    const program = flowToProgram(nodes, edges);
    const addStmt = program.statements.find(
      (s) => s.type === "TransformStatement" && s.identifier === "add"
    );
    expect(addStmt?.type).toBe("TransformStatement");
    if (addStmt?.type !== "TransformStatement") return;
    expect(addStmt.arguments.map((a) => (a.type === "Identifier" ? a.name : ""))).toEqual([
      "output_out1",
      "n4b",
    ]);

    const interpreter = new Interpreter();
    const { results, errors } = await interpreter.execute(program);
    expect(errors).toHaveLength(0);

    expect(numericValue(results.get("output_out1")!)).toBe(1);
    expect(numericValue(results.get("output_out2")!)).toBe(5);
  });
});

describe("flowToProgram order operators", () => {
  test("orden-menor-mayor emits order_asc with a single argument", () => {
    const nodes = [
      { id: "grp", type: "arrayClose", position: { x: 0, y: 0 }, data: {} },
      operator("ord", "orden-menor-mayor"),
    ] as DataflowNode[];
    const edges: Edge[] = [
      { id: "e1", source: "grp", target: "ord", sourceHandle: "out", targetHandle: "a" },
    ];
    const program = flowToProgram(nodes, edges);
    const stmt = program.statements.find(
      (s) => s.type === "TransformStatement" && s.identifier === "ord"
    );
    expect(stmt?.type).toBe("TransformStatement");
    if (stmt?.type !== "TransformStatement") return;
    expect(stmt.operation).toBe("order_asc");
    expect(stmt.arguments).toHaveLength(2);
    expect(stmt.arguments[0]?.type).toBe("Identifier");
    if (stmt.arguments[0]?.type === "Identifier") {
      expect(stmt.arguments[0].name).toBe("grp");
    }
    expect(stmt.arguments[1]?.type).toBe("CriteriaLiteral");
  });
});

describe("flowToProgram filter operators", () => {
  test("filtrar-general emits filter with group and criteria arguments", () => {
    const nodes = [
      { id: "grp", type: "arrayClose", position: { x: 0, y: 0 }, data: {} },
      {
        id: "crit",
        type: "source",
        position: { x: 0, y: 0 },
        data: {
          variant: "criteria",
          yoloClass: "large",
          properties: ["size"],
          values: { size: "grande" },
        },
      },
      {
        id: "flt",
        type: "operator",
        position: { x: 0, y: 0 },
        data: { operator: "filtrar-general" },
      },
    ] as DataflowNode[];
    const edges: Edge[] = [
      { id: "e1", source: "grp", target: "flt", sourceHandle: "out", targetHandle: "a" },
      { id: "e2", source: "crit", target: "flt", sourceHandle: "out", targetHandle: "b" },
    ];
    const program = flowToProgram(nodes, edges);
    const stmt = program.statements.find(
      (s) => s.type === "TransformStatement" && s.identifier === "flt"
    );
    expect(stmt?.type).toBe("TransformStatement");
    if (stmt?.type !== "TransformStatement") return;
    expect(stmt.operation).toBe("filter");
    expect(stmt.arguments.map((a) => (a.type === "Identifier" ? a.name : ""))).toEqual([
      "grp",
      "crit",
    ]);
  });
});

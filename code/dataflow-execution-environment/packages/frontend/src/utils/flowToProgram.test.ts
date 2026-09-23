import { describe, expect, test } from "bun:test";
import { Interpreter, isBag, type RuntimeValue } from "@dataflow/interpreter";
import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/node/types";
import { flowToProgram, resolveFlowSourceId } from "./flowToProgram";
import { spawnActionForYoloClass } from "@/data/yoloDeckCatalog";

/** Un resultado numérico: una bolsa de una sola entrada. */
function numericValue(value: RuntimeValue): number {
  if (!isBag(value) || value.entries.length !== 1) {
    throw new Error(`no es un número: ${JSON.stringify(value.kind)}`);
  }
  return Number(value.entries[0].quantity.valueOf());
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

describe("las cuatro cartas de orden", () => {
  /** El criterio que emite la carta del mazo con esa clase YOLO. */
  function criterionOf(yoloClass: string) {
    const spawn = spawnActionForYoloClass(yoloClass);
    if (spawn?.kind !== "operator") throw new Error(`${yoloClass} no es un operador`);

    const nodes = [
      { id: "grp", type: "arrayClose", position: { x: 0, y: 0 }, data: {} },
      {
        id: "ord",
        type: "operator",
        position: { x: 0, y: 0 },
        // Lo que guarda el nodo al crear la carta, criterio incluido.
        data: { operator: spawn.operator, criterio: spawn.criterio },
      },
    ] as DataflowNode[];
    const edges: Edge[] = [
      { id: "e1", source: "grp", target: "ord", sourceHandle: "out", targetHandle: "a" },
    ];

    const program = flowToProgram(nodes, edges);
    const criterion = program.statements.find(
      (s) => s.type === "SourceStatement" && s.identifier === "ord__criterio"
    );
    if (criterion?.type !== "SourceStatement" || criterion.value.type !== "CriterionLiteral") {
      throw new Error("falta el source del criterio de orden");
    }
    return criterion.value.values;
  }

  test("las del mazo traen su criterio: dos por cantidad y dos por tamaño", () => {
    expect(criterionOf("ascending")).toEqual({ quantity: "asc" });
    expect(criterionOf("descending")).toEqual({ quantity: "desc" });
    expect(criterionOf("smallest_to_largest")).toEqual({
      size: ["pequeño", "mediano", "grande"],
    });
    expect(criterionOf("largest_to_smallest")).toEqual({
      size: ["grande", "mediano", "pequeño"],
    });
  });

  test("ninguna pareja emite lo mismo", () => {
    const criterios = [
      "ascending",
      "descending",
      "smallest_to_largest",
      "largest_to_smallest",
    ].map((carta) => JSON.stringify(criterionOf(carta)));

    expect(new Set(criterios).size).toBe(4);
  });
});

describe("flowToProgram order operators", () => {
  function orderProgram(op: "orden-menor-mayor" | "orden-mayor-menor") {
    const nodes = [
      { id: "grp", type: "arrayClose", position: { x: 0, y: 0 }, data: {} },
      operator("ord", op),
    ] as DataflowNode[];
    const edges: Edge[] = [
      { id: "e1", source: "grp", target: "ord", sourceHandle: "out", targetHandle: "a" },
    ];
    return flowToProgram(nodes, edges);
  }

  test("una sola operación `order`, con el criterio en su propio source", () => {
    const program = orderProgram("orden-menor-mayor");

    const stmt = program.statements.find(
      (s) => s.type === "TransformStatement" && s.identifier === "ord"
    );
    expect(stmt?.type).toBe("TransformStatement");
    if (stmt?.type !== "TransformStatement") return;

    // Los argumentos son solo identificadores (§5.1).
    expect(stmt.operation).toBe("order");
    expect(stmt.arguments.map((a) => a.name)).toEqual(["grp", "ord__criterio"]);

    const criterion = program.statements.find(
      (s) => s.type === "SourceStatement" && s.identifier === "ord__criterio"
    );
    if (criterion?.type !== "SourceStatement" || criterion.value.type !== "CriterionLiteral") {
      throw new Error("falta el source del criterio de orden");
    }
    expect(criterion.value.sourceType).toBe("order");
    expect(criterion.value.values).toEqual({ quantity: "asc" });
  });

  test("el sentido viaja en el criterio, no en el nombre de la operación", () => {
    const program = orderProgram("orden-mayor-menor");

    const stmt = program.statements.find(
      (s) => s.type === "TransformStatement" && s.identifier === "ord"
    );
    if (stmt?.type !== "TransformStatement") throw new Error("falta el transform");
    expect(stmt.operation).toBe("order");

    const criterion = program.statements.find(
      (s) => s.type === "SourceStatement" && s.identifier === "ord__criterio"
    );
    if (criterion?.type !== "SourceStatement" || criterion.value.type !== "CriterionLiteral") {
      throw new Error("falta el source del criterio de orden");
    }
    expect(criterion.value.values).toEqual({ quantity: "desc" });
  });

  test("con un criterio de secuencia, el sentido inverso la invierte", () => {
    const nodes = [
      { id: "grp", type: "arrayClose", position: { x: 0, y: 0 }, data: {} },
      {
        id: "ord",
        type: "operator",
        position: { x: 0, y: 0 },
        data: {
          operator: "orden-mayor-menor",
          criterio: { property: "size", sequence: ["pequeño", "mediano", "grande"] },
        },
      },
    ] as DataflowNode[];
    const edges: Edge[] = [
      { id: "e1", source: "grp", target: "ord", sourceHandle: "out", targetHandle: "a" },
    ];

    const program = flowToProgram(nodes, edges);
    const criterion = program.statements.find(
      (s) => s.type === "SourceStatement" && s.identifier === "ord__criterio"
    );
    if (criterion?.type !== "SourceStatement" || criterion.value.type !== "CriterionLiteral") {
      throw new Error("falta el source del criterio de orden");
    }
    expect(criterion.value.values).toEqual({ size: ["grande", "mediano", "pequeño"] });
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
    expect(stmt.arguments.map((a) => a.name)).toEqual(["grp", "crit"]);

    // La carta de criterio declara su subtipo: es un criterio de filtro.
    const criterion = program.statements.find(
      (s) => s.type === "SourceStatement" && s.identifier === "crit"
    );
    if (criterion?.type !== "SourceStatement" || criterion.value.type !== "CriterionLiteral") {
      throw new Error("falta el source del criterio");
    }
    expect(criterion.value.sourceType).toBe("filter");
    expect(criterion.value.values).toEqual({ size: "grande" });
  });
});

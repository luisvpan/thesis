import { describe, expect, test } from "bun:test";
import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/node/types";
import { canvasNodeIdOf, describeNode } from "./describeNode";

const nodes = [
  { id: "n7", type: "source", position: { x: 0, y: 0 }, data: { variant: "number", value: 7 } },
  {
    id: "estrella",
    type: "source",
    position: { x: 0, y: 600 },
    data: { variant: "shape", shape: "estrella", size: "pequeña", color: "azul", yoloClass: "sm" },
  },
  {
    id: "circulo",
    type: "source",
    position: { x: 0, y: 1200 },
    data: { variant: "shape", shape: "circulo", size: "grande", color: "rojo", yoloClass: "lg" },
  },
  { id: "manzana", type: "source", position: { x: 0, y: 1800 }, data: { variant: "food", food: "manzana" } },
  { id: "cubo", type: "source", position: { x: 0, y: 2400 }, data: { variant: "montessori", color: "rojo" } },
  { id: "tapa", type: "source", position: { x: 0, y: 3000 }, data: { variant: "cap", color: "azul" } },
  {
    id: "criterio",
    type: "source",
    position: { x: 0, y: 3600 },
    data: { variant: "criteria", properties: ["size"], values: { size: "grande" } },
  },
  {
    id: "orden",
    type: "operator",
    position: { x: 0, y: 4200 },
    data: { operator: "orden-mayor-menor", criterio: { property: "size", sequence: [] } },
  },
  { id: "suma", type: "operator", position: { x: 0, y: 4800 }, data: { operator: "adicion" } },
  { id: "salida", type: "programOutput", position: { x: 0, y: 5400 }, data: {} },
] as DataflowNode[];

const name = (id: string, edges: Edge[] = []) => describeNode(id, nodes, edges);

describe("describeNode", () => {
  test("las cartas de datos se nombran con género y número concordados", () => {
    expect(name("n7")).toBe("el número 7");
    expect(name("estrella")).toBe("la estrella pequeña azul");
    expect(name("circulo")).toBe("el círculo grande rojo");
    expect(name("manzana")).toBe("la manzana");
    expect(name("cubo")).toBe("el cubo rojo");
    expect(name("tapa")).toBe("la tapa azul");
  });

  test("los operadores dicen qué hacen, y el de orden por qué ordena", () => {
    expect(name("suma")).toBe("la suma");
    expect(name("orden")).toBe("el orden por tamaño de mayor a menor");
  });

  test("el criterio dice de qué es", () => {
    expect(name("criterio")).toBe("el criterio de tamaño");
  });

  test("los dos ids sintéticos apuntan a la carta que se ve", () => {
    // El sink de una salida y el criterio implícito de una carta de orden: ese
    // criterio no es una carta, así que señala al operador que lo declara.
    expect(name("output_salida")).toBe("la salida");
    expect(name("orden__criterio")).toBe("el orden por tamaño de mayor a menor");

    expect(canvasNodeIdOf("output_salida")).toBe("salida");
    expect(canvasNodeIdOf("orden__criterio")).toBe("orden");
    expect(canvasNodeIdOf("n7")).toBe("n7");
  });

  test("un id que no está en la mesa no se nombra", () => {
    expect(name("fantasma")).toBeNull();
    expect(describeNode(undefined, nodes, [])).toBeNull();
  });
});

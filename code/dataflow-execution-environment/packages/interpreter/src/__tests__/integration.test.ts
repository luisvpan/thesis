// Las doce operaciones y los ejemplos de la especificación, de punta a punta.

import { describe, expect, test } from "bun:test";
import { Interpreter } from "../index";
import type { BooleanValue } from "../runtime/types";
import { dataLiteral, numberLiteral, only, pairs } from "./helpers";

const num = numberLiteral;

async function run(program: string) {
  const result = await new Interpreter().execute(program);
  expect(result.errors.map((error) => error.message)).toEqual([]);
  return result.results;
}

describe("Aritmética (§3.1)", () => {
  test("suma exacta sobre ℚ: 1/2 + 1/3 = 5/6 (§5.3)", async () => {
    const results = await run(`
      source half = ${num("1/2")};
      source third = ${num("1/3")};
      transform total = sum(half, third);
      sink output = total;
    `);
    expect(only(results.get("output")!)).toBe("5/6");
  });

  test("los atributos forman parte de la identidad: sum no agrupa lo que difiere (§5.3)", async () => {
    const results = await run(`
      source sedan = ${dataLiteral("concreto", "vehicle", "car", 2, { doors: "4" })};
      source coupe = ${dataLiteral("concreto", "vehicle", "car", 1, { doors: "2" })};
      transform vehicles = sum(sedan, coupe);
      sink output = vehicles;
    `);
    expect(pairs(results.get("output")!)).toEqual(["car:2", "car:1"]);
  });

  test("un grupo declara una bolsa de varias entradas y sum la agrega", async () => {
    const results = await run(`
      source items = [
        ${dataLiteral("concreto", "comida", "uva", 5, { color: "morado" })},
        ${dataLiteral("concreto", "comida", "uva", 3, { color: "morado" })},
        ${dataLiteral("concreto", "comida", "manzana", 2, { color: "rojo" })}
      ];
      transform total = sum(items);
      sink result = total;
    `);
    expect(pairs(results.get("result")!)).toEqual(["uva:8", "manzana:2"]);
  });

  test("expresión compuesta: (3 + 2) × (10 − 6) = 20", async () => {
    const results = await run(`
      source a = ${num(3)};
      source b = ${num(2)};
      source c = ${num(10)};
      source d = ${num(6)};
      transform add = sum(a, b);
      transform difference = substract(c, d);
      transform product = multiply(add, difference);
      sink result = product;
    `);
    expect(only(results.get("result")!)).toBe("20");
  });

  test("escalado: 2.5 × 3 = 7.5 (§5.3)", async () => {
    const results = await run(`
      source large_star = ${dataLiteral("pictorico", "shape", "star", 2.5, { size: "large" })};
      source scale_factor = ${num(3)};
      transform scaled_stars = multiply(large_star, scale_factor);
      sink final_render = scaled_stars;
    `);
    expect(pairs(results.get("final_render")!)).toEqual(["star:15/2"]);
  });

  test("división exacta", async () => {
    const results = await run(`
      source a = ${dataLiteral("pictorico", "forma", "circulo", 12, { size: "mediano" })};
      source b = ${num(4)};
      transform quotient = divide(a, b);
      sink result = quotient;
    `);
    expect(pairs(results.get("result")!)).toEqual(["circulo:3"]);
  });
});

describe("Comparación (§3.2)", () => {
  test("less_than y greater_than filtran por la cantidad total", async () => {
    const results = await run(`
      source frutas = [
        ${dataLiteral("concreto", "comida", "manzana", 2)},
        ${dataLiteral("concreto", "comida", "pera", 5)}
      ];
      source tres = ${num(3)};
      transform pocas = less_than(frutas, tres);
      transform muchas = greater_than(frutas, tres);
      sink a = pocas;
      sink b = muchas;
    `);
    expect(pairs(results.get("a")!)).toEqual(["manzana:2"]);
    expect(pairs(results.get("b")!)).toEqual(["pera:5"]);
  });

  test("compare ignora las cantidades 0: 3 manzanas − 3 manzanas ≈ nulo", async () => {
    const results = await run(`
      source tres = ${dataLiteral("concreto", "comida", "manzana", 3)};
      source vacio = ;
      transform resta = substract(tres, tres);
      transform iguales = compare(resta, vacio);
      sink result = iguales;
    `);
    expect((results.get("result") as BooleanValue).value).toBe(true);
  });

  test("compare ignora orden y agrupación", async () => {
    const results = await run(`
      source a = [${dataLiteral("concreto", "comida", "manzana", 1)}, ${dataLiteral("concreto", "comida", "manzana", 2)}];
      source b = ${dataLiteral("concreto", "comida", "manzana", 3)};
      transform iguales = compare(a, b);
      sink result = iguales;
    `);
    expect((results.get("result") as BooleanValue).value).toBe(true);
  });
});

describe("Criterios (§1.3, §5.3)", () => {
  const frutas = `[
    ${dataLiteral("concreto", "food", "apple", 3)},
    ${dataLiteral("concreto", "food", "pear", 1)}
  ]`;

  test("criterio de filtro: valor único sobre una propiedad de identidad", async () => {
    const results = await run(`
      source fruits = ${frutas};
      source only_apples = {"sourceType": "filter", "properties": ["subtype"], "subtype": "apple"};
      transform apples = filter(fruits, only_apples);
      sink out_apples = apples;
    `);
    expect(pairs(results.get("out_apples")!)).toEqual(["apple:3"]);
  });

  test("criterio de orden: dirección sobre la cantidad", async () => {
    const results = await run(`
      source fruits = ${frutas};
      source by_qty = {"sourceType": "order", "properties": ["quantity"], "quantity": "asc"};
      transform sorted = order(fruits, by_qty);
      sink out_sorted = sorted;
    `);
    expect(pairs(results.get("out_sorted")!)).toEqual(["pear:1", "apple:3"]);
  });

  test("criterio de orden: secuencia explícita de valores", async () => {
    const results = await run(`
      source estrellas = [
        ${dataLiteral("pictorico", "forma", "estrella", 1, { size: "grande" })},
        ${dataLiteral("pictorico", "forma", "estrella", 2, { size: "pequena" })},
        ${dataLiteral("pictorico", "forma", "estrella", 3, { size: "mediana" })}
      ];
      source por_tamano = {"sourceType": "order", "properties": ["size"], "size": ["pequena", "mediana", "grande"]};
      transform ordenadas = order(estrellas, por_tamano);
      sink result = ordenadas;
    `);
    expect(pairs(results.get("result")!)).toEqual(["estrella:2", "estrella:3", "estrella:1"]);
  });

  test("varios criterios de filtro son una disyunción", async () => {
    const results = await run(`
      source frutas = [
        ${dataLiteral("concreto", "comida", "manzana", 2)},
        ${dataLiteral("concreto", "comida", "pera", 3)},
        ${dataLiteral("concreto", "comida", "uva", 1)}
      ];
      source manzanas = {"sourceType": "filter", "properties": ["subtype"], "subtype": "manzana"};
      source uvas = {"sourceType": "filter", "properties": ["subtype"], "subtype": "uva"};
      transform elegidas = filter(frutas, manzanas, uvas);
      sink result = elegidas;
    `);
    expect(pairs(results.get("result")!)).toEqual(["manzana:2", "uva:1"]);
  });
});

describe("Acceso y agregación (§3.5, §3.6)", () => {
  test("first y last leen el orden vigente", async () => {
    const results = await run(`
      source frutas = [
        ${dataLiteral("concreto", "comida", "manzana", 2)},
        ${dataLiteral("concreto", "comida", "pera", 3)},
        ${dataLiteral("concreto", "comida", "uva", 1)}
      ];
      transform primera = first(frutas);
      transform ultima = last(frutas);
      sink a = primera;
      sink b = ultima;
    `);
    expect(pairs(results.get("a")!)).toEqual(["manzana:2"]);
    expect(pairs(results.get("b")!)).toEqual(["uva:1"]);
  });

  test("count totaliza y su resultado sirve de escalar", async () => {
    const results = await run(`
      source frutas = [
        ${dataLiteral("concreto", "comida", "manzana", 2)},
        ${dataLiteral("concreto", "comida", "pera", 3)}
      ];
      source uvas = ${dataLiteral("concreto", "comida", "uva", 1)};
      transform cuantas = count(frutas);
      transform escaladas = multiply(uvas, cuantas);
      sink total = cuantas;
      sink result = escaladas;
    `);
    expect(only(results.get("total")!)).toBe("5");
    expect(pairs(results.get("result")!)).toEqual(["uva:5"]);
  });
});

describe("Programas parciales (§2.5)", () => {
  test("un transform sin operación evalúa a nulo", async () => {
    const results = await run(`
      transform incomplete_calc = ;
      sink active_output = incomplete_calc;
    `);
    expect(pairs(results.get("active_output")!)).toEqual([]);
  });

  test("un criterio a medio declarar no invalida el programa", async () => {
    const results = await run(`
      source frutas = ${dataLiteral("concreto", "comida", "manzana", 2)};
      source criterio = ;
      transform filtradas = filter(frutas, criterio);
      sink result = filtradas;
    `);
    expect(pairs(results.get("result")!)).toEqual(["manzana:2"]);
  });
});

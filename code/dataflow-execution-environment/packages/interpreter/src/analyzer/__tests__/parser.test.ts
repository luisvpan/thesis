// §5 Gramática

import { describe, expect, test } from "bun:test";
import type { DataLiteral, GroupLiteral, Program, TransformStatement } from "../ast";
import { parseToAst } from "../../serializer";

function parse(input: string): { ast: Program | null; errors: string[] } {
  const { ast, errors } = parseToAst(input);
  return { ast, errors: errors.map((error) => error.message) };
}

function sourceLiteral(ast: Program, index = 0) {
  const stmt = ast.statements[index];
  if (stmt.type !== "SourceStatement") throw new Error("no es un source");
  return stmt.value;
}

describe("Literales de datos", () => {
  test.each(["abstracto", "pictorico", "concreto"])("admite la categoría %s", (category) => {
    const result = parse(
      `source x = {"sourceType": "data", "category": "${category}", "type": "t", "subtype": "s", "quantity": 5};`
    );
    expect(result.errors).toHaveLength(0);
    expect((sourceLiteral(result.ast!) as DataLiteral).category).toBe(category);
  });

  test("rechaza una categoría fuera de las tres", () => {
    const result = parse(
      `source x = {"sourceType": "data", "category": "liquido", "type": "t", "subtype": "s", "quantity": 5};`
    );
    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("el literal racional admite entero, decimal y fracción", () => {
    for (const [text, expected] of [
      ["5", "5"],
      ["3.14", "3.14"],
      ["1/3", "1/3"],
      ["-2", "-2"],
    ]) {
      const result = parse(
        `source x = {"sourceType": "data", "category": "abstracto", "type": "numero", "subtype": "racional", "quantity": ${text}};`
      );
      expect(result.errors).toHaveLength(0);
      expect((sourceLiteral(result.ast!) as DataLiteral).quantity).toBe(expected);
    }
  });

  test("sin sourceType se interpreta como dato", () => {
    const result = parse(`source x = {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 1};`);
    expect(result.errors).toHaveLength(0);
    expect(sourceLiteral(result.ast!)?.type).toBe("DataLiteral");
  });

  test("rechaza un número suelto", () => {
    expect(parse("source x = 5;").errors.length).toBeGreaterThan(0);
    expect(parse("source x = -2.5;").errors.length).toBeGreaterThan(0);
  });

  test("rechaza un texto suelto", () => {
    expect(parse('source x = "hola";').errors.length).toBeGreaterThan(0);
  });
});

describe("Grupos", () => {
  test("un grupo reúne objetos de datos", () => {
    const result = parse(`
      source g = [
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 1},
        {"category": "concreto", "type": "comida", "subtype": "pera", "quantity": 2}
      ];
    `);
    expect(result.errors).toHaveLength(0);
    expect((sourceLiteral(result.ast!) as GroupLiteral).elements).toHaveLength(2);
  });

  test("el grupo vacío es válido", () => {
    const result = parse("source g = [];");
    expect(result.errors).toHaveLength(0);
    expect((sourceLiteral(result.ast!) as GroupLiteral).elements).toHaveLength(0);
  });

  test("un grupo no puede mezclar datos y criterios", () => {
    const result = parse(`
      source g = [
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 1},
        {"sourceType": "filter", "properties": ["subtype"], "subtype": "uva"}
      ];
    `);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("un source no referencia otros nodos", () => {
    expect(parse("source z = [a, b];").errors.length).toBeGreaterThan(0);
  });
});

describe("Criterios", () => {
  test("declara su subtipo en sourceType", () => {
    const result = parse(`source c = {"sourceType": "filter", "properties": ["size"], "size": "grande"};`);
    expect(result.errors).toHaveLength(0);

    const literal = sourceLiteral(result.ast!);
    expect(literal?.type).toBe("CriterionLiteral");
    expect(literal).toMatchObject({ sourceType: "filter", properties: ["size"] });
  });

  test("rechaza un sourceType que no existe", () => {
    const result = parse(`source c = {"sourceType": "criteria", "properties": ["size"], "size": "grande"};`);
    expect(result.errors.length).toBeGreaterThan(0);
  });
});

describe("Transform y sink", () => {
  test("la operación es un identificador cualquiera", () => {
    const result = parse("transform t = cualquier_cosa(a, b);");
    expect(result.errors).toHaveLength(0);
    expect((result.ast!.statements[0] as TransformStatement).operation).toBe("cualquier_cosa");
  });

  test("los argumentos son solo identificadores", () => {
    expect(parse('transform t = sum(a, {"category": "concreto"});').errors.length).toBeGreaterThan(0);
  });

  test("los nodos incompletos se toleran", () => {
    const result = parse("source x = ; transform t = ; sink s = ;");
    expect(result.errors).toHaveLength(0);
    expect(result.ast!.statements).toHaveLength(3);
    expect(sourceLiteral(result.ast!)).toBeUndefined();
    expect((result.ast!.statements[1] as TransformStatement).operation).toBeUndefined();
  });

  test("admite comentarios", () => {
    const result = parse("/* un comentario */ sink s = x;");
    expect(result.errors).toHaveLength(0);
    expect(result.ast!.statements).toHaveLength(1);
  });
});

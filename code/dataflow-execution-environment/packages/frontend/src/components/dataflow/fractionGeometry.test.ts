import { describe, expect, test } from "bun:test";
import { divisionLines, isDrawableFraction, wedgeClipPath } from "./fractionGeometry";

/** Los vértices del recorte, sin el centro. */
function rim(path: string): [number, number][] {
  const inside = path.slice("polygon(".length, -1);
  return inside
    .split(", ")
    .slice(1)
    .map((point) => {
      const [x, y] = point.split(" ");
      return [Number.parseFloat(x), Number.parseFloat(y)] as [number, number];
    });
}

describe("isDrawableFraction", () => {
  test("acepta una fracción propia con pocas regiones", () => {
    expect(isDrawableFraction(4, 7)).toBe(true);
    expect(isDrawableFraction(1, 2)).toBe(true);
  });

  test("rechaza lo que no es una parte de algo", () => {
    expect(isDrawableFraction(0, 7)).toBe(false); // nada presente
    expect(isDrawableFraction(7, 7)).toBe(false); // entero: no hay que dividirlo
    expect(isDrawableFraction(-1, 2)).toBe(false); // negativo
    expect(isDrawableFraction(1.5, 3)).toBe(false);
  });

  test("rechaza los denominadores que no se pueden leer", () => {
    expect(isDrawableFraction(4, 13)).toBe(false);
  });
});

describe("wedgeClipPath", () => {
  test("arranca a las 12 y avanza en sentido horario", () => {
    const points = rim(wedgeClipPath(1, 4));
    const [first] = points;
    const last = points[points.length - 1];

    // Arriba, centrado.
    expect(first[0]).toBeCloseTo(50, 1);
    expect(first[1]).toBeLessThan(0);
    // Un cuarto de vuelta: a la derecha, a media altura.
    expect(last[0]).toBeGreaterThan(100);
    expect(last[1]).toBeCloseTo(50, 1);
  });

  test("muestrea el arco para que la cuña cubra las esquinas", () => {
    // Tres cuartos son 270°: con pasos de 30° hacen falta vértices intermedios,
    // o el polígono cortaría por la cuerda y se comería las esquinas.
    expect(rim(wedgeClipPath(3, 4)).length).toBeGreaterThan(3);
  });

  test("a mayor porción, más área presente", () => {
    expect(rim(wedgeClipPath(1, 8)).length).toBeLessThan(rim(wedgeClipPath(7, 8)).length);
  });
});

describe("divisionLines", () => {
  test("hay una línea por región", () => {
    expect(divisionLines(4, 7)).toHaveLength(7);
  });

  test("entera la que bordea lo presente, punteada la que cruza el fantasma", () => {
    // 4 de 7: las líneas 0..4 cierran las cuatro regiones llenas; las demás, no.
    expect(divisionLines(4, 7).map((line) => line.solid)).toEqual([
      true,
      true,
      true,
      true,
      true,
      false,
      false,
    ]);
  });

  test("las líneas salen del centro hacia el borde", () => {
    for (const line of divisionLines(1, 6)) {
      const radius = Math.hypot(line.x - 50, line.y - 50);
      expect(radius).toBeGreaterThan(70); // media diagonal de la caja
    }
  });
});

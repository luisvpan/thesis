import { describe, expect, test } from "bun:test";
import { loopEdgeProgress, offsetPointAbove } from "./edgePathWalker";

describe("edgePathWalker", () => {
  test("loopEdgeProgress wraps at 1", () => {
    expect(loopEdgeProgress(0, 1000)).toBe(0);
    expect(loopEdgeProgress(500, 1000)).toBe(0.5);
    expect(loopEdgeProgress(1000, 1000)).toBe(0);
    expect(loopEdgeProgress(2500, 1000)).toBe(0.5);
  });

  test("offsetPointAbove shifts upward for horizontal tangent", () => {
    const p = offsetPointAbove(10, 20, 1, 0, 8);
    expect(p.x).toBe(10);
    expect(p.y).toBeLessThan(20);
  });

  test("offsetPointAbove at midpoint of horizontal segment", () => {
    const above = offsetPointAbove(50, 0, 1, 0, 10);
    expect(above.x).toBe(50);
    expect(above.y).toBe(-10);
  });
});

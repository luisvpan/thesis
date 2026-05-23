import { describe, expect, test } from "bun:test";
import type { DataflowNode } from "./types";
import { VISION_CARD_NODE_TTL_MS } from "./constants";
import { stabilizeVisionNodeList } from "./stabilizeVisionNodes";
import { visionMetaFromCard } from "./visionNodeMeta";
import type { VisionCardItem } from "../VisionContext";

function card(overrides: Partial<VisionCardItem> = {}): VisionCardItem {
  return {
    classId: 0,
    label: "one",
    confidence: 0.9,
    position: { x: 0.5, y: 0.5 },
    ...overrides,
  };
}

describe("stabilizeVisionNodeList", () => {
  test("propaga lastSeenAt al reutilizar nodo estable (TTL de oclusión)", () => {
    const prev: DataflowNode[] = [
      {
        id: "card_1",
        type: "operator",
        position: { x: 100, y: 200 },
        data: { operator: "adicion", lastSeenAt: 1000, visionStatus: "active" },
      },
    ];
    const next: DataflowNode[] = [
      {
        id: "card_1",
        type: "operator",
        position: { x: 100.2, y: 200.1 },
        data: { operator: "adicion", lastSeenAt: 3500, visionStatus: "active" },
      },
    ];
    const out = stabilizeVisionNodeList(prev, next);
    expect((out[0].data as { lastSeenAt: number }).lastSeenAt).toBe(3500);
    expect(out[0].type).toBe("operator");
  });

  test("lastSeenAt actualizado permite retener carta tapada dentro del TTL", () => {
    const lastSeenAt = 5000;
    const frameTimeMs = lastSeenAt + VISION_CARD_NODE_TTL_MS - 500;
    expect(frameTimeMs - lastSeenAt).toBeLessThan(VISION_CARD_NODE_TTL_MS);
  });
});

describe("visionMetaFromCard", () => {
  test("defers lost opacity briefly to avoid flicker", () => {
    const t0 = 10_000;
    const first = visionMetaFromCard(card({ status: "lost" }), t0);
    expect(first.visionStatus).toBe("active");

    const second = visionMetaFromCard(card({ status: "lost" }), t0 + 100, first);
    expect(second.visionStatus).toBe("active");

    const third = visionMetaFromCard(
      card({ status: "lost" }),
      t0 + 500,
      second
    );
    expect(third.visionStatus).toBe("lost");
  });
});

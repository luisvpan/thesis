import { describe, expect, test } from "bun:test";
import {
  isVisionTrackDegraded,
  visionNodePresentationFields,
  visionNodeZIndex,
  VISION_ACTIVE_Z_INDEX,
  VISION_DIMMED_Z_INDEX,
  VISION_DIMMED_OPACITY,
  withVisionNodeChrome,
} from "./visionNodePresentation";

describe("visionNodePresentation", () => {
  test("degraded statuses are lost and stale", () => {
    expect(isVisionTrackDegraded("lost")).toBe(true);
    expect(isVisionTrackDegraded("stale")).toBe(true);
    expect(isVisionTrackDegraded("active")).toBe(false);
    expect(isVisionTrackDegraded(undefined)).toBe(false);
  });

  test("z-index is lower when degraded", () => {
    expect(visionNodeZIndex("active")).toBe(VISION_ACTIVE_Z_INDEX);
    expect(visionNodeZIndex("lost")).toBe(VISION_DIMMED_Z_INDEX);
  });

  test("dimmed nodes get opacity style", () => {
    expect(visionNodePresentationFields({ visionStatus: "lost" }).style).toEqual({
      opacity: VISION_DIMMED_OPACITY,
    });
    expect(visionNodePresentationFields({ visionStatus: "active" }).style).toBeUndefined();
  });

  test("withVisionNodeChrome sets draggable only when enabled", () => {
    const draggable = withVisionNodeChrome(
      { id: "card_1", type: "source", position: { x: 0, y: 0 }, data: {} },
      { visionStatus: "active" },
      true
    );
    const fixed = withVisionNodeChrome(
      { id: "card_2", type: "source", position: { x: 0, y: 0 }, data: {} },
      { visionStatus: "lost" },
      false
    );
    expect(draggable.draggable).toBe(true);
    expect(fixed.draggable).toBe(false);
    expect(fixed.zIndex).toBe(VISION_DIMMED_Z_INDEX);
  });
});

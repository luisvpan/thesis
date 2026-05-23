/**
 * Visual shape variants for flow handles (form only; colors are state-driven).
 */
export type FlowHandleVariant =
  | "input-out"
  | "operator-in-a"
  | "operator-in-b"
  | "operator-out"
  | "sink-in"
  | "sink-out"
  | "zone-open-triangle"
  | "zone-close-triangle";

/** clip-path / border-radius classes per variant */
export function getHandleVariantShapeClass(variant: FlowHandleVariant | undefined): string {
  if (!variant) return "";

  switch (variant) {
    case "input-out":
      return ""; // falls through to HandleKind shape in ClickableHandle
    case "operator-in-a":
      return "flow-handle-hexagon !rounded-none";
    case "operator-in-b":
      return "flow-handle-diamond !rounded-none";
    case "operator-out":
      return "flow-handle-chevron !rounded-none";
    case "sink-in":
      return "flow-handle-sink-in !rounded-sm";
    case "sink-out":
      return "!rounded-md";
    case "zone-open-triangle":
      return "flow-handle-triangle-open !rounded-none";
    case "zone-close-triangle":
      return "flow-handle-triangle-close !rounded-none";
    default:
      return "";
  }
}

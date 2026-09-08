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
  | "zone-close-triangle"
  | "zone-close-out";

const FLOW_HANDLE_ROUNDED = "!rounded-md";

/** clip-path / border-radius classes per variant */
export function getHandleVariantShapeClass(variant: FlowHandleVariant | undefined): string {
  if (!variant) return "";

  switch (variant) {
    case "input-out":
      return ""; // falls through to HandleKind shape in ClickableHandle
    case "operator-in-a":
    case "operator-in-b":
      return `flow-handle-hexagon ${FLOW_HANDLE_ROUNDED}`;
    case "operator-out":
    case "zone-close-out":
    case "sink-in":
      return `!rounded-full ${FLOW_HANDLE_ROUNDED}`;
    case "sink-out":
      return `flow-handle-arrow ${FLOW_HANDLE_ROUNDED}`;
    case "zone-open-triangle":
      return `flow-handle-triangle-open ${FLOW_HANDLE_ROUNDED}`;
    case "zone-close-triangle":
      return `flow-handle-triangle-close ${FLOW_HANDLE_ROUNDED}`;
    default:
      return "";
  }
}

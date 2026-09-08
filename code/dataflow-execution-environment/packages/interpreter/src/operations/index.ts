import type { RuntimeValue } from "../runtime/types";
import type { Operation } from "../analyzer/ast";
import { sum, substract, multiply, divide } from "./arithmetic";
import { lessThan, greaterThan } from "./comparison";
import { orderAsc, orderDesc } from "./ordering";
import { filter } from "./filtering";
import { first, last } from "./accessor";
import { count } from "./aggregation";
import { compare } from "./equality";
import { RuntimeError } from "../runtime/errors";

type OperationFn = (args: RuntimeValue[]) => RuntimeValue;

const OPERATION_REGISTRY: Record<Operation, OperationFn> = {
  sum,
  substract,
  multiply,
  divide,
  less_than: lessThan,
  greater_than: greaterThan,
  order_asc: orderAsc,
  order_desc: orderDesc,
  filter,
  first,
  last,
  count,
  compare,
};

export function executeOperation(
  op: Operation,
  args: RuntimeValue[]
): RuntimeValue {
  const fn = OPERATION_REGISTRY[op];
  if (!fn) {
    throw new RuntimeError("UNKNOWN_OPERATION", `Unknown operation: ${op}`);
  }
  return fn(args);
}

// Re-export individual operations for direct use
export { sum, substract, multiply, divide } from "./arithmetic";
export { lessThan, greaterThan } from "./comparison";
export { orderAsc, orderDesc } from "./ordering";
export { filter } from "./filtering";
export { first, last } from "./accessor";
export { count } from "./aggregation";
export { compare } from "./equality";

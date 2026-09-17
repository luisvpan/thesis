import { RuntimeError } from "../runtime/errors";
import type { RuntimeValue } from "../runtime/types";
import { count } from "./aggregation";
import { divide, multiply, substract, sum } from "./arithmetic";
import { first, last } from "./accessor";
import { compare } from "./equality";
import { filter } from "./filtering";
import { greaterThan, lessThan } from "./comparison";
import { order } from "./ordering";
import { isOperation, type Operation } from "./signatures";

type OperationFn = (args: RuntimeValue[]) => RuntimeValue;

const OPERATION_REGISTRY: Record<Operation, OperationFn> = {
  sum,
  substract,
  multiply,
  divide,
  less_than: lessThan,
  greater_than: greaterThan,
  compare,
  order,
  filter,
  first,
  last,
  count,
};

/**
 * Despacha una operación. Que el nombre pertenezca al conjunto reconocido lo
 * garantiza la pasada estática (§4.2.4); el chequeo de aquí es la red de
 * seguridad para quien llame al intérprete sin pasar por ella.
 */
export function executeOperation(operation: string, args: RuntimeValue[]): RuntimeValue {
  if (!isOperation(operation)) {
    throw new RuntimeError("UNKNOWN_OPERATION", `La operación '${operation}' no existe`);
  }

  return OPERATION_REGISTRY[operation](args);
}

export { sum, substract, multiply, divide } from "./arithmetic";
export { lessThan, greaterThan } from "./comparison";
export { order } from "./ordering";
export { filter } from "./filtering";
export { first, last } from "./accessor";
export { count } from "./aggregation";
export { compare } from "./equality";
export {
  OPERATIONS,
  SIGNATURES,
  describeArity,
  isOperation,
  parameterAt,
  type Operation,
  type OperationSignature,
  type ParameterSpec,
} from "./signatures";

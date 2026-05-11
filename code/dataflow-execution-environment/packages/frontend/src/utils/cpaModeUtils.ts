import type { ResultViewMode } from '@/components/dataflow/dataflowResultCpa';
import type { SourceFlowNodeData } from '@/components/dataflow/SourceFlowNode';
import type { OperatorType } from '@/types/card-types';

/** CPA de la carta-fuente, alineado con el lenguaje (concreto / pictorico / abstracto). */
export function sourceCpaCategory(data: SourceFlowNodeData): ResultViewMode {
  if (data.variant === 'number') return 'abstracto';
  if (data.variant === 'shape') return 'pictorico';
  return 'concreto';
}

export function isSourceBlockedByCpaMode(
  data: SourceFlowNodeData,
  mode: ResultViewMode
): boolean {
  return sourceCpaCategory(data) !== mode;
}

function isFilterOperator(operator: OperatorType): boolean {
  return operator.startsWith('filtrar-');
}

/**
 * - Multiplicación y división: solo en abstracto.
 * - Filtrar: solo en concreto y pictórico (no en abstracto).
 */
export function isOperatorBlockedByCpaMode(
  operator: OperatorType,
  mode: ResultViewMode
): boolean {
  if (operator === 'multiplicacion' || operator === 'division') {
    return mode !== 'abstracto';
  }
  if (isFilterOperator(operator)) {
    return mode === 'abstracto';
  }
  return false;
}
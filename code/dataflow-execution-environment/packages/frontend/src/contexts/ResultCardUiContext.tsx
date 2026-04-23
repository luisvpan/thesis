import { createContext, useContext, type ReactNode } from 'react';
import type { ResultViewMode } from '@/components/dataflow/dataflowResultCpa';

export type ResultCardUiState = {
  viewMode: ResultViewMode;
  hasExecuted: boolean;
  /** Cuando es true, cada operador puede mostrar su resultado local según CPA. */
  showOperatorResults: boolean;
};

const ResultCardUiContext = createContext<ResultCardUiState | null>(null);

export function ResultCardUiProvider({
  viewMode,
  hasExecuted,
  showOperatorResults,
  children,
}: ResultCardUiState & { children: ReactNode }) {
  return (
    <ResultCardUiContext.Provider
      value={{ viewMode, hasExecuted, showOperatorResults }}
    >
      {children}
    </ResultCardUiContext.Provider>
  );
}

export function useResultCardUi(): ResultCardUiState {
  const ctx = useContext(ResultCardUiContext);
  if (!ctx) {
    throw new Error('useResultCardUi must be used within ResultCardUiProvider');
  }
  return ctx;
}

import { createContext, useContext, type ReactNode } from 'react';
import type { ResultViewMode } from '@/components/dataflow/dataflowResultCpa';

export type ResultCardUiState = {
  viewMode: ResultViewMode;
  hasExecuted: boolean;
  /** Muestra miniaturas animadas sobre conectores hacia operadores/sinks. */
  showFlowResults: boolean;
};

const ResultCardUiContext = createContext<ResultCardUiState | null>(null);

export function ResultCardUiProvider({
  viewMode,
  hasExecuted,
  showFlowResults,
  children,
}: ResultCardUiState & { children: ReactNode }) {
  return (
    <ResultCardUiContext.Provider value={{ viewMode, hasExecuted, showFlowResults }}>
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

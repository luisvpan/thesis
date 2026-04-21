import { createContext, useContext, type ReactNode } from 'react';
import type { ResultViewMode } from '@/components/dataflow/dataflowResultCpa';

export type ResultCardUiState = {
  viewMode: ResultViewMode;
  hasExecuted: boolean;
};

const ResultCardUiContext = createContext<ResultCardUiState | null>(null);

export function ResultCardUiProvider({
  viewMode,
  hasExecuted,
  children,
}: ResultCardUiState & { children: ReactNode }) {
  return (
    <ResultCardUiContext.Provider value={{ viewMode, hasExecuted }}>
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

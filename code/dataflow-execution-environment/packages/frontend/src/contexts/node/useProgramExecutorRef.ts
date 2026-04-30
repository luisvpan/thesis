import { useEffect, useRef, type MutableRefObject } from "react";
import {
  createProgramExecutor,
  type ProgramExecutor,
} from "@/services/executeProgram";

export function useProgramExecutorRef(): MutableRefObject<ProgramExecutor | null> {
  const executorRef = useRef<ProgramExecutor | null>(null);

  useEffect(() => {
    executorRef.current = createProgramExecutor();
    console.log("[NodeProvider] Interpreter created");

    return () => {
      executorRef.current?.reset();
      executorRef.current = null;
      console.log("[NodeProvider] Interpreter destroyed");
    };
  }, []);

  return executorRef;
}

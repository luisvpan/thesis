import { useEffect, useRef, type MutableRefObject } from "react";
import {
  createProgramExecutor,
  type ProgramExecutor,
} from "@/services/executeProgram";
import { logger } from "@/lib/logger";

export function useProgramExecutorRef(): MutableRefObject<ProgramExecutor | null> {
  const executorRef = useRef<ProgramExecutor | null>(null);

  useEffect(() => {
    executorRef.current = createProgramExecutor();
    logger.nodeProvider.debug("Interpreter created");

    return () => {
      executorRef.current?.reset();
      executorRef.current = null;
      logger.nodeProvider.debug("Interpreter destroyed");
    };
  }, []);

  return executorRef;
}

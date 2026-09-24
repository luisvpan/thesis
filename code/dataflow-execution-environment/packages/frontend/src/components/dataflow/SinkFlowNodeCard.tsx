import type { ReactNode } from "react";
import type { NodeErrorMark } from "@/contexts/node/errorMarks";
import { NodeErrorBadge } from "./NodeErrorBadge";

type SinkFlowNodeCardProps = {
  /** Texto principal (descripción semántica, error, valor, etc.) */
  headerRight: ReactNode;
  /** Cubos / iconos / franja visual debajo del título */
  resultVisual?: ReactNode;
  /** Botón de "Escuchar" (u otra acción), mostrado junto al texto del resultado. */
  actionButton?: ReactNode;
  /** Distintivo de error de esta salida; al pulsarlo cuenta qué pasó (§4). */
  errorMark?: NodeErrorMark;
  className?: string;
};

/**
 * Layout de carta sink: texto del resultado (con su botón de acción al lado)
 * justo arriba de la representación gráfica de los elementos.
 */
export function SinkFlowNodeCard({
  headerRight,
  resultVisual,
  actionButton,
  errorMark,
  className = "",
}: SinkFlowNodeCardProps) {
  return (
    <div
      className={`relative flex w-full flex-col gap-3 px-3 pb-3 text-white ${className}`}
    >
      <NodeErrorBadge mark={errorMark} />
      <div className="ml-36 flex w-full items-end gap-3 text-md font-semibold leading-snug text-teal-200">
        {actionButton}
      </div>
      <div className="w-full space-y-4">
        <div className="min-w-0 flex-1">{headerRight}</div>
        {resultVisual}
      </div>
    </div>
  );
}

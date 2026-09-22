import type { ReactNode } from "react";

type SinkFlowNodeCardProps = {
  /** Texto principal (descripción semántica, error, valor, etc.) */
  headerRight: ReactNode;
  /** Cubos / iconos / franja visual debajo del título */
  resultVisual?: ReactNode;
  /** Botón de "Escuchar" (u otra acción), mostrado junto al texto del resultado. */
  actionButton?: ReactNode;
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
  className = "",
}: SinkFlowNodeCardProps) {
  return (
    <div
      className={`flex w-full flex-col gap-3 px-3 pb-3 text-white ${className}`}
    >
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

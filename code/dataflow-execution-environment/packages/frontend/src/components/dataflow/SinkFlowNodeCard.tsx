import type { ReactNode } from 'react';

type SinkFlowNodeCardProps = {
  /** Texto principal (descripción semántica, error, valor, etc.) */
  headerRight: ReactNode;
  /** Cubos / iconos / franja visual debajo del título */
  resultVisual?: ReactNode;
  className?: string;
};

/**
 * Layout de carta sink: etiqueta «Salida» a la izquierda, título a la derecha,
 * representación del resultado en la parte inferior.
 */
export function SinkFlowNodeCard({
  headerRight,
  resultVisual,
  className = '',
}: SinkFlowNodeCardProps) {
  return (
    <div
      className={`flex min-h-36 w-full flex-col justify-between gap-3 p-3 text-white ${className}`}
    >
      <div className="grid grid-cols-2 text-md font-semibold leading-snug text-teal-200">
        <div className="col-span-1"></div>
        <div className="col-span-1">{headerRight}</div>
      </div>
      {resultVisual ? <div className=" w-full">{resultVisual}</div> : null}
    </div>
  );
}

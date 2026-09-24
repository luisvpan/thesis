import type { ReactNode } from 'react';
import type { CardCategory } from '@/types/card-types';
import type { NodeErrorMark } from '@/contexts/node/errorMarks';
import { NodeErrorBadge } from './NodeErrorBadge';

type FlowCardFamily = 'input' | 'transformation' | 'sink';

/** Atenuación del resto del camino de una salida que falló: leve, como en Scratch. */
const FLOW_DIMMED_CLASS = 'opacity-70';

type FlowNodeCardProps = {
  family: FlowCardFamily;
  title: string;
  content: ReactNode;
  subtitle?: string;
  className?: string;
  /** Tipo de carta tangible; permite estilos distintos por categoría sin cambiar la API del nodo. */
  cardCategory?: CardCategory;
  /** Aviso encima de la fila del título (label), sin alterar el resto de la carta. */
  topNotice?: ReactNode;
  /** Si false, solo se muestra el contenido (p. ej. dígitos fusionados sin repetir el título). */
  showHeader?: boolean;
  /** Papel de la carta en el error de una salida, si lo tiene (§4). */
  errorMark?: NodeErrorMark;
};

export function FlowNodeCard({
  family: _family,
  title,
  content,
  subtitle: _subtitle,
  cardCategory: _cardCategory,
  className = '',
  topNotice,
  showHeader = true,
  errorMark,
}: FlowNodeCardProps) {
  const dimmed = errorMark?.role === 'flow' ? FLOW_DIMMED_CLASS : '';

  return (
    <div className={`relative p-3 text-white ${dimmed} ${className}`}>
      <NodeErrorBadge mark={errorMark} />
      {topNotice}
      {showHeader ? (
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="truncate text-xs font-semibold uppercase tracking-wide text-slate-300">
            {title}-{content}
          </span>
        </div>
      ) : (
        null
      )}
    </div>
  );
}

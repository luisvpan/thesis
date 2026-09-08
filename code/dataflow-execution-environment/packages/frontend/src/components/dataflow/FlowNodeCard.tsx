import type { ReactNode } from 'react';
import type { CardCategory } from '@/types/card-types';

type FlowCardFamily = 'input' | 'transformation' | 'sink';

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
}: FlowNodeCardProps) {
  return (
    <div className={` p-3 text-white ${className}`}>
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

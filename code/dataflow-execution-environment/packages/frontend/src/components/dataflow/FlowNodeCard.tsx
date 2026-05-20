import type { ReactNode } from 'react';

type FlowCardFamily = 'input' | 'transformation' | 'sink';

type FlowNodeCardProps = {
  family: FlowCardFamily;
  title: string;
  content: ReactNode;
  subtitle?: string;
  className?: string;
  /** Aviso encima de la fila del título (label), sin alterar el resto de la carta. */
  topNotice?: ReactNode;
};

export function FlowNodeCard({
  family: _family,
  title,
  content,
  subtitle: _subtitle,
  className = '',
  topNotice,
}: FlowNodeCardProps) {
  return (
    <div className={`min-h-48 w-48 p-3 text-white ${className}`}>
      {topNotice}
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="truncate text-xs font-semibold uppercase tracking-wide text-slate-300">
          {title}-{content}
        </span>
      </div>
    </div>
  );
}

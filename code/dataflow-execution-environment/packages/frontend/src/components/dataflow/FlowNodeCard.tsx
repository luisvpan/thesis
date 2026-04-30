import type { ReactNode } from 'react';

type FlowCardFamily = 'input' | 'transformation' | 'sink';

type FlowNodeCardProps = {
  family: FlowCardFamily;
  title: string;
  content: ReactNode;
  subtitle?: string;
  className?: string;
};

function familyBadge(family: FlowCardFamily): string {
  if (family === 'input') return 'Input';
  if (family === 'transformation') return 'Transformation';
  return 'Sink';
}

export function FlowNodeCard({ family, title, content, subtitle, className = '' }: FlowNodeCardProps) {
  return (
    <div
      className={`h-48 w-48 rounded-2xl border-2 border-slate-500 bg-slate-900/90 p-3 text-white shadow-xl ${className}`}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="truncate text-xs font-semibold uppercase tracking-wide text-slate-300">{title}</span>
        <span className="rounded-md border border-slate-600 bg-slate-800 px-1.5 py-0.5 text-[10px] font-semibold text-slate-200">
          {familyBadge(family)}
        </span>
      </div>

      <div className="flex h-28 items-center justify-center rounded-xl border border-slate-600 bg-slate-950 px-2 text-center">
        {content}
      </div>

      <div className="mt-2 truncate text-center text-xs text-slate-300">{subtitle ?? ''}</div>
    </div>
  );
}

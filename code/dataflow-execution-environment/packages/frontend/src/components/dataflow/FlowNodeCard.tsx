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
  console.log("content", content);
  console.log("title", title);
  return (
    <div
      className={`h-48 w-48   p-3 text-white ${className}`}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="truncate text-xs font-semibold uppercase tracking-wide text-slate-300">{title}-{content}</span>
      </div>
    </div>
  );
}

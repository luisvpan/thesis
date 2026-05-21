import type { SourceCardCategory } from './sourceNodeLayout';
import { FlowNodeCard } from '../FlowNodeCard';
import { sourceMain, sourceTitle } from './sourceNodeLabels';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type SourceCardNodeBodyProps = {
  cardCategory: SourceCardCategory;
  data: SourceFlowNodeData;
  cardStyleClassName?: string;
};

/**
 * Cuerpo visual compartido por todas las cartas fuente.
 * Mismo diseño actual; `cardCategory` permite ramificar estilos por tipo más adelante.
 */
export function SourceCardNodeBody({ cardCategory, data, cardStyleClassName }: SourceCardNodeBodyProps) {
  const subtitle = data.variant === 'number' ? data.visionSubtitle : undefined;

  return (
    <FlowNodeCard
      family="input"
      cardCategory={cardCategory}
      title={sourceTitle(data)}
      content={<span className="text-xs font-black text-slate-100">{sourceMain(data)}</span>}
      subtitle={subtitle}
      className={cardStyleClassName}
    />
  );
}

import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type MontessoriSourceData = Extract<SourceFlowNodeData, { variant: 'montessori' }>;

export function MontessoriSourceFlowNode({ data }: { data: MontessoriSourceData }) {
  return <SourceCardNodeBody cardCategory="montessori" data={data} />;
}

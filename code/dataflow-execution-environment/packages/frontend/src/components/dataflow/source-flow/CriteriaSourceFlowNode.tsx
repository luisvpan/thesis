import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type CriteriaSourceData = Extract<SourceFlowNodeData, { variant: 'criteria' }>;

export function CriteriaSourceFlowNode({ data }: { data: CriteriaSourceData }) {
  return <SourceCardNodeBody cardCategory="criteria" data={data} />;
}

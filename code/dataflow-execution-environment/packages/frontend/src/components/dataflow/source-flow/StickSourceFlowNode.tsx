import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type StickSourceData = Extract<SourceFlowNodeData, { variant: 'stick' }>;

export function StickSourceFlowNode({ data }: { data: StickSourceData }) {
  return <SourceCardNodeBody cardCategory="stick" data={data} />;
}

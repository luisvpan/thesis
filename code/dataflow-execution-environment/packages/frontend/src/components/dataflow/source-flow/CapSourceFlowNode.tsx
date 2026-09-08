import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type CapSourceData = Extract<SourceFlowNodeData, { variant: 'cap' }>;

export function CapSourceFlowNode({ data }: { data: CapSourceData }) {
  return <SourceCardNodeBody cardCategory="cap" data={data} />;
}

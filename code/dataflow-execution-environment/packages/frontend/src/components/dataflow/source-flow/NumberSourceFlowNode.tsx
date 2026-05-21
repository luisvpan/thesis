import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type NumberSourceData = Extract<SourceFlowNodeData, { variant: 'number' }>;

export function NumberSourceFlowNode({ data }: { data: NumberSourceData }) {
  return <SourceCardNodeBody cardCategory="number" data={data} />;
}

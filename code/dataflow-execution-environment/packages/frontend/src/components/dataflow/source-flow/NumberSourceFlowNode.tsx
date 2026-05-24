import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';
import { isNumberMergeHead } from '@/utils/numberTouchMerge';

type NumberSourceData = Extract<SourceFlowNodeData, { variant: 'number' }>;

export function NumberSourceFlowNode({
  data,
  nodeId,
}: {
  data: NumberSourceData;
  nodeId: string;
}) {
  return (
    <SourceCardNodeBody
      cardCategory="number"
      data={data}
      showHeader={isNumberMergeHead(nodeId, data)}
    />
  );
}

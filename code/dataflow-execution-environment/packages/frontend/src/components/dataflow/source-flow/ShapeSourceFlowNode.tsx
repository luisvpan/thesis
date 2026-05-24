import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type ShapeSourceData = Extract<SourceFlowNodeData, { variant: 'shape' }>;

export function ShapeSourceFlowNode({ data }: { data: ShapeSourceData }) {
  return <SourceCardNodeBody cardCategory="shape" data={data} />;
}

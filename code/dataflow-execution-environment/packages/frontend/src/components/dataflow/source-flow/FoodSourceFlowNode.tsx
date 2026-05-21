import { SourceCardNodeBody } from './SourceCardNodeBody';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type FoodSourceData = Extract<SourceFlowNodeData, { variant: 'food' }>;

export function FoodSourceFlowNode({ data }: { data: FoodSourceData }) {
  return <SourceCardNodeBody cardCategory="food" data={data} />;
}

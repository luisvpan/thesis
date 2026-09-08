import type { SourceFlowNodeData } from '../SourceFlowNode';
import { NumberSourceFlowNode } from './NumberSourceFlowNode';
import { ShapeSourceFlowNode } from './ShapeSourceFlowNode';
import { MontessoriSourceFlowNode } from './MontessoriSourceFlowNode';
import { CapSourceFlowNode } from './CapSourceFlowNode';
import { StickSourceFlowNode } from './StickSourceFlowNode';
import { FoodSourceFlowNode } from './FoodSourceFlowNode';
import { CriteriaSourceFlowNode } from './CriteriaSourceFlowNode';
export function renderSourceFlowNodeBody(data: SourceFlowNodeData, nodeId: string) {
  switch (data.variant) {
    case 'number':
      return <NumberSourceFlowNode data={data} nodeId={nodeId} />;
    case 'shape':
      return <ShapeSourceFlowNode data={data} />;
    case 'montessori':
      return <MontessoriSourceFlowNode data={data} />;
    case 'cap':
      return <CapSourceFlowNode data={data} />;
    case 'stick':
      return <StickSourceFlowNode data={data} />;
    case 'food':
      return <FoodSourceFlowNode data={data} />;
    case 'criteria':
      return <CriteriaSourceFlowNode data={data} />;
  }
}

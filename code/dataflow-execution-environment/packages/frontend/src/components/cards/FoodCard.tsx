import { CardBase } from './CardBase';
import type { FoodType } from '@/types/card-types';

interface FoodCardProps {
  food: FoodType;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

interface FoodConfig {
  emoji: string;
  name: string;
}

const foodConfig: Record<FoodType, FoodConfig> = {
  'manzana': { emoji: '🍎', name: 'Manzana' },
  'hamburguesa': { emoji: '🍔', name: 'Hamburguesa' },
  'uvas': { emoji: '🍇', name: 'Uvas' },
  'pasta': { emoji: '🍝', name: 'Pasta' },
  'peras': { emoji: '🍐', name: 'Peras' }
};

export function FoodCard({ food, isDraggable = false, onClick, size = 'medium' }: FoodCardProps) {
  const config = foodConfig[food];

  return (
    <CardBase 
      borderColor="#f97316" // Naranja
      isDraggable={isDraggable}
      onClick={onClick}
      size={size}
      cardType="ENTRADA"
      cardName={config.name}
    >
      {config.emoji}
    </CardBase>
  );
}

import { CardBase } from './CardBase';
import type { FoodType } from '@/types/card-types';
import { FOOD_EMOJI } from '@/data/foodEmoji';

interface FoodCardProps {
  food: FoodType;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

const foodNames: Record<FoodType, string> = {
  manzana: 'Manzana',
  hamburguesa: 'Hamburguesa',
  uvas: 'Uvas',
  pasta: 'Pasta',
  peras: 'Peras',
  naranja: 'Naranja',
};

export function FoodCard({ food, isDraggable = false, onClick, size = 'medium' }: FoodCardProps) {
  return (
    <CardBase 
      borderColor="#f97316" // Naranja
      isDraggable={isDraggable}
      onClick={onClick}
      size={size}
      cardType="ENTRADA"
      cardName={foodNames[food]}
    >
      {FOOD_EMOJI[food]}
    </CardBase>
  );
}

import { CardBase } from './CardBase';
import type { CarColor } from '@/types/card-types';

interface CarCardProps {
  color: CarColor;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

interface CarConfig {
  emoji: string;
  name: string;
}

const carConfig: Record<CarColor, CarConfig> = {
  'rojo': { emoji: '🚗', name: 'Carro Rojo' },
  'negro': { emoji: '🚙', name: 'Carro Negro' },
  'amarillo': { emoji: '🚖', name: 'Carro Amarillo' },
  'azul-oscuro': { emoji: '🚓', name: 'Carro Azul Oscuro' },
  'gris': { emoji: '🚐', name: 'Carro Gris' }
};

export function CarCard({ color, isDraggable = false, onClick, size = 'medium' }: CarCardProps) {
  const config = carConfig[color];

  return (
    <CardBase 
      borderColor="#000000" // Negro
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
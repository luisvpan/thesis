import { CardBase } from './CardBase';
import type { AnimalType } from '@/types/card-types';

interface AnimalCardProps {
  animal: AnimalType;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

interface AnimalConfig {
  emoji: string;
  name: string;
}

const animalConfig: Record<AnimalType, AnimalConfig> = {
  'gato': { emoji: '🐱', name: 'Gato' },
  'perro': { emoji: '🐶', name: 'Perro' },
  'tortuga': { emoji: '🐢', name: 'Tortuga' },
  'elefante': { emoji: '🐘', name: 'Elefante' },
  'jirafa': { emoji: '🦒', name: 'Jirafa' }
};

export function AnimalCard({ animal, isDraggable = false, onClick, size = 'medium' }: AnimalCardProps) {
  const config = animalConfig[animal];

  return (
    <CardBase 
      borderColor="#10b981" // Verde
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
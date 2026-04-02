import { CardBase } from './CardBase';
import type { PersonGender, PersonAge } from '@/types/card-types';

interface PersonCardProps {
  gender: PersonGender;
  age: PersonAge;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

interface PersonConfig {
  emoji: string;
  name: string;
}

// Configuración de personas: género × edad
const personConfig: Record<PersonGender, Record<PersonAge, PersonConfig>> = {
  'mujer': {
    'bebe': { emoji: '👶', name: 'Bebé Mujer' },
    'niño': { emoji: '👧', name: 'Niña' },
    'joven': { emoji: '👩', name: 'Joven Mujer' },
    'adulto': { emoji: '👩‍🦳', name: 'Adulta' }
  },
  'hombre': {
    'bebe': { emoji: '👶', name: 'Bebé Hombre' },
    'niño': { emoji: '👦', name: 'Niño' },
    'joven': { emoji: '👨', name: 'Joven Hombre' },
    'adulto': { emoji: '👨‍🦳', name: 'Adulto' }
  }
};

export function PersonCard({ gender, age, isDraggable = false, onClick, size = 'medium' }: PersonCardProps) {
  const config = personConfig[gender][age];

  return (
    <CardBase 
      borderColor="#a855f7" // Morado
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
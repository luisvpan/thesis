import { CardBase } from './CardBase';

interface NumberCardProps {
  value: number;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

const numberNames: Record<number, string> = {
  0: 'Cero',
  1: 'Uno',
  2: 'Dos',
  3: 'Tres',
  4: 'Cuatro',
  5: 'Cinco',
  6: 'Seis',
  7: 'Siete',
  8: 'Ocho',
  9: 'Nueve'
};

export function NumberCard({ value, isDraggable = false, onClick, size = 'medium' }: NumberCardProps) {
  return (
    <CardBase 
      borderColor="#3b82f6" // Azul
      isDraggable={isDraggable}
      onClick={onClick}
      size={size}
      cardType="ENTRADA"
      cardName={numberNames[value]}
    >
      {value}
    </CardBase>
  );
}
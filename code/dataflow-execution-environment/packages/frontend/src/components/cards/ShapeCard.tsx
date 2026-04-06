import { CardBase } from './CardBase';
import type { ShapeType, ShapeSize, ShapeColor } from '@/types/card-types';

interface ShapeCardProps {
  shape: ShapeType;
  size: ShapeSize;
  color: ShapeColor;
  isDraggable?: boolean;
  onClick?: () => void;
  cardSize?: 'small' | 'medium' | 'large';
}

const shapeColors: Record<ShapeColor, { hex: string; gradient: string; name: string }> = {
  'rojo': { hex: '#ef4444', gradient: 'linear-gradient(to bottom right, #ef4444, #dc2626)', name: 'Rojo' },
  'azul': { hex: '#3b82f6', gradient: 'linear-gradient(to bottom right, #3b82f6, #2563eb)', name: 'Azul' },
  'amarillo': { hex: '#eab308', gradient: 'linear-gradient(to bottom right, #eab308, #ca8a04)', name: 'Amarillo' },
  'verde': { hex: '#10b981', gradient: 'linear-gradient(to bottom right, #10b981, #059669)', name: 'Verde' }
};

const shapeNames: Record<ShapeType, string> = {
  'triangulo': 'Triángulo',
  'cuadrado': 'Cuadrado',
  'rectangulo': 'Rectángulo',
  'rombo': 'Rombo',
  'circulo': 'Círculo',
  'estrella': 'Estrella',
  'trapecio': 'Trapecio'
};

const sizeNames: Record<ShapeSize, string> = {
  'pequeña': 'Pequeño',
  'mediana': 'Mediano',
  'grande': 'Grande'
};

// Formas que tienen variantes de tamaño y color
const shapesWithVariants: ShapeType[] = ['triangulo', 'circulo', 'cuadrado'];

export function ShapeCard({ shape, size, color, isDraggable = false, onClick, cardSize = 'medium' }: ShapeCardProps) {
  const hasVariants = shapesWithVariants.includes(shape);
  const shapeName = hasVariants ? `${shapeNames[shape]} ${sizeNames[size]}` : shapeNames[shape];
  const shapeColor = shapeColors[color];

  const renderShape = () => {
    const sizes = {
      small: { pequeña: 24, mediana: 32, grande: 40 },
      medium: { pequeña: 32, mediana: 48, grande: 64 },
      large: { pequeña: 48, mediana: 64, grande: 80 }
    };

    const shapeSize = sizes[cardSize][size];

    switch (shape) {
      case 'triangulo':
        return (
          <svg width={shapeSize} height={shapeSize} viewBox="0 0 100 100">
            <defs>
              <linearGradient id={`grad-tri-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: shapeColor.hex, stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: shapeColor.hex, stopOpacity: 0.7 }} />
              </linearGradient>
            </defs>
            <polygon points="50,10 90,90 10,90" fill={`url(#grad-tri-${color})`} stroke={shapeColor.hex} strokeWidth="3" />
          </svg>
        );

      case 'cuadrado':
        return (
          <svg width={shapeSize} height={shapeSize} viewBox="0 0 100 100">
            <defs>
              <linearGradient id={`grad-sq-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: shapeColor.hex, stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: shapeColor.hex, stopOpacity: 0.7 }} />
              </linearGradient>
            </defs>
            <rect x="10" y="10" width="80" height="80" fill={`url(#grad-sq-${color})`} stroke={shapeColor.hex} strokeWidth="3" />
          </svg>
        );

      case 'circulo':
        return (
          <svg width={shapeSize} height={shapeSize} viewBox="0 0 100 100">
            <defs>
              <linearGradient id={`grad-circ-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: shapeColor.hex, stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: shapeColor.hex, stopOpacity: 0.7 }} />
              </linearGradient>
            </defs>
            <circle cx="50" cy="50" r="40" fill={`url(#grad-circ-${color})`} stroke={shapeColor.hex} strokeWidth="3" />
          </svg>
        );

      case 'rectangulo':
        return (
          <svg width={shapeSize * 1.5} height={shapeSize} viewBox="0 0 150 100">
            <defs>
              <linearGradient id={`grad-rect-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: shapeColor.hex, stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: shapeColor.hex, stopOpacity: 0.7 }} />
              </linearGradient>
            </defs>
            <rect x="10" y="25" width="130" height="50" fill={`url(#grad-rect-${color})`} stroke={shapeColor.hex} strokeWidth="3" />
          </svg>
        );

      case 'rombo':
        return (
          <svg width={shapeSize} height={shapeSize} viewBox="0 0 100 100">
            <defs>
              <linearGradient id={`grad-rhom-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: shapeColor.hex, stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: shapeColor.hex, stopOpacity: 0.7 }} />
              </linearGradient>
            </defs>
            <polygon points="50,10 90,50 50,90 10,50" fill={`url(#grad-rhom-${color})`} stroke={shapeColor.hex} strokeWidth="3" />
          </svg>
        );

      case 'estrella':
        return (
          <svg width={shapeSize} height={shapeSize} viewBox="0 0 100 100">
            <defs>
              <linearGradient id={`grad-star-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: shapeColor.hex, stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: shapeColor.hex, stopOpacity: 0.7 }} />
              </linearGradient>
            </defs>
            <polygon points="50,10 61,40 92,40 67,59 78,90 50,70 22,90 33,59 8,40 39,40" fill={`url(#grad-star-${color})`} stroke={shapeColor.hex} strokeWidth="3" />
          </svg>
        );

      case 'trapecio':
        return (
          <svg width={shapeSize} height={shapeSize} viewBox="0 0 100 100">
            <defs>
              <linearGradient id={`grad-trap-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: shapeColor.hex, stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: shapeColor.hex, stopOpacity: 0.7 }} />
              </linearGradient>
            </defs>
            <polygon points="30,20 70,20 90,80 10,80" fill={`url(#grad-trap-${color})`} stroke={shapeColor.hex} strokeWidth="3" />
          </svg>
        );

      default:
        return null;
    }
  };

  return (
    <CardBase
      borderColor="#ec4899" // Rosado
      isDraggable={isDraggable}
      onClick={onClick}
      size={cardSize}
      cardType="ENTRADA"
      cardName={shapeName}
    >
      {renderShape()}
    </CardBase>
  );
}
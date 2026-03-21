import { motion } from 'motion/react';
import type { ReactNode } from 'react';

interface CardBaseProps {
  children: ReactNode;
  borderColor: string;
  backgroundColor?: string;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
  cardType?: 'OPERADOR' | 'ENTRADA'; // Tipo de carta
  cardName?: string; // Nuevo: nombre de la carta
  iconColor?: string; // Color del ícono/símbolo
}

export function CardBase({ 
  children, 
  borderColor, 
  backgroundColor = 'white',
  isDraggable = false,
  onClick,
  size = 'medium',
  cardType,
  cardName,
  iconColor
}: CardBaseProps) {
  const sizeClasses = {
    small: 'w-24 h-32',
    medium: 'w-32 h-44',
    large: 'w-40 h-56'
  };

  const iconContainerSizes = {
    small: 'w-16 h-16',
    medium: 'w-24 h-24',
    large: 'w-32 h-32'
  };

  const iconSizes = {
    small: 'text-5xl',
    medium: 'text-7xl',
    large: 'text-8xl'
  };

  const nameSizes = {
    small: 'text-sm',
    medium: 'text-base',
    large: 'text-xl'
  };

  const idSizes = {
    small: 'text-[10px]',
    medium: 'text-xs',
    large: 'text-sm'
  };

  // Convertir gradiente a color sólido
  const getSolidColor = (gradient: string): string => {
    const colorMatch = gradient.match(/#[0-9a-fA-F]{6}/);
    return colorMatch ? colorMatch[0] : '#3b82f6';
  };

  const solidColor = getSolidColor(borderColor);
  const displayIconColor = iconColor || solidColor;

  return (
    <motion.div
      drag={isDraggable}
      dragMomentum={false}
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className={`${isDraggable ? 'cursor-grab active:cursor-grabbing' : 'cursor-pointer'}`}
    >
      <div 
        className={`${sizeClasses[size]} rounded-2xl relative`}
        style={{ 
          background: borderColor,
        }}
      >
        {/* Contenedor principal blanco */}
        <div className="absolute inset-[2px] rounded-2xl bg-white flex flex-col items-center justify-center p-3">
          
          {/* Tipo de carta centrado arriba */}
          {cardType && (
            <div 
              className={`absolute top-2 w-full text-center ${idSizes[size]} font-bold text-black`}
              style={{ 
                textShadow: '1px 1px 0px rgba(0, 0, 0, 0.1)'
              }}
            >
              {cardType}
            </div>
          )}

          {/* Ícono/Símbolo central con sombra larga */}
          <div className="flex-1 flex items-center justify-center mb-2">
            <div 
              className={`${iconContainerSizes[size]} rounded-2xl flex items-center justify-center relative z-10`}
              style={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                boxShadow: '4px 4px 0px rgba(0, 0, 0, 0.1)'
              }}
            >
              {/* Wrapper para manejar tanto texto/emoji como elementos SVG */}
              <div 
                className={`flex items-center justify-center ${iconSizes[size]} font-black`}
                style={{ 
                  color: displayIconColor,
                  textShadow: '2px 2px 0px rgba(0, 0, 0, 0.1)'
                }}
              >
                {children}
              </div>
            </div>
          </div>

          {/* Nombre de la carta */}
          {cardName && (
            <div 
              className={`${nameSizes[size]} font-black text-center px-2 text-black`}
              style={{ 
                textShadow: '1px 1px 0px rgba(0, 0, 0, 0.1)'
              }}
            >
              {cardName}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
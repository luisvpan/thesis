import { motion } from 'motion/react';

interface NumberCardProps {
  value: number;
  /** Si se define, sustituye el nombre del número (p. ej. clase YOLO no numérica). */
  subtitle?: string;
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
  9: 'Nueve',
};

const sizeClasses = {
  small: 'min-w-[5.5rem] min-h-[4.5rem] px-2 py-2',
  medium: 'min-w-[7rem] min-h-[5.5rem] px-3 py-3',
  large: 'min-w-[9rem] min-h-[7rem] px-4 py-4',
};

const labelClasses = {
  small: 'text-sm',
  medium: 'text-base',
  large: 'text-lg',
};

export function NumberCard({
  value,
  subtitle,
  isDraggable = false,
  onClick,
  size = 'medium',
}: NumberCardProps) {
  const label = subtitle ?? numberNames[value] ?? String(value);

  return (
    <motion.div
      drag={isDraggable}
      dragMomentum={false}
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className={isDraggable ? 'cursor-grab active:cursor-grabbing' : onClick ? 'cursor-pointer' : ''}
    >
      <div
        className={`${sizeClasses[size]} rounded-lg flex items-center justify-center border-2 border-[#3b82f6] bg-transparent`}
      >
        <span className={`${labelClasses[size]} font-semibold text-center text-blue-400`}>{label}</span>
      </div>
    </motion.div>
  );
}

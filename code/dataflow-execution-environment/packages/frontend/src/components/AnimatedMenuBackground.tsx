import { motion } from 'motion/react';

const WAVES_PATTERN = `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 20 Q 15 10, 30 20 T 60 20' stroke='%234299e1' stroke-width='2' fill='none'/%3E%3Cpath d='M0 40 Q 15 30, 30 40 T 60 40' stroke='%234299e1' stroke-width='2' fill='none'/%3E%3C/svg%3E")`;

export function AnimatedMenuBackground() {
  return (
    <motion.div
      className="absolute inset-0 opacity-25 pointer-events-none"
      style={{
        backgroundImage: WAVES_PATTERN,
        backgroundSize: '60px 60px',
      }}
      animate={{
        backgroundPositionX: ['0px', '60px'],
      }}
      transition={{
        duration: 3,
        repeat: Infinity,
        ease: 'linear',
      }}
    />
  );
}

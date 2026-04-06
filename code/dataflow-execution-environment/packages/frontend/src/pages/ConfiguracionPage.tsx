import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowLeft, LockOpen, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { AnimatedMenuBackground } from '@/components/AnimatedMenuBackground';

const OPTIONS = [
  { id: 'unlock', name: 'Desbloquear Niveles', icon: LockOpen },
  { id: 'clear', name: 'Borrar Datos', icon: Trash2 },
] as const;

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.1 },
  },
};

const item = {
  hidden: { opacity: 0, x: -20 },
  show: { opacity: 1, x: 0 },
};

export default function ConfiguracionPage() {
  const navigate = useNavigate();
  const [unlocked, setUnlocked] = useState(false);
  const [cleared, setCleared] = useState(false);

  const handleUnlock = () => {
    setUnlocked(true);
    // Aquí podrías guardar en localStorage o contexto
    setTimeout(() => setUnlocked(false), 2000);
  };

  const handleClear = () => {
    if (window.confirm('¿Borrar todos los datos guardados? Esta acción no se puede deshacer.')) {
      setCleared(true);
      // Aquí podrías limpiar localStorage / contexto
      setTimeout(() => setCleared(false), 2000);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-sky-300 via-sky-200 to-emerald-100 flex flex-col items-center p-6 md:p-8">
      <AnimatedMenuBackground />
      <div className="relative z-10 w-full max-w-md flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8 w-full"
        >
          <button
            type="button"
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            Volver
          </button>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-4xl md:text-5xl font-black text-slate-800 mb-2 text-center"
        >
          Configuración
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-slate-600 mb-10 text-center"
        >
          Ajustes del juego
        </motion.p>

        <motion.ul
          variants={container}
          initial="hidden"
          animate="show"
          className="flex flex-col gap-4 w-full"
        >
        <AnimatePresence>
          {OPTIONS.map((opt, i) => {
            const Icon = opt.icon;
            const isUnlock = opt.id === 'unlock';
            const isClear = opt.id === 'clear';
            const done = (isUnlock && unlocked) || (isClear && cleared);
            return (
              <motion.li key={opt.id} variants={item}>
                <motion.button
                  type="button"
                  whileHover={{ scale: 1.02, x: 8 }}
                  whileTap={{ scale: 0.98 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 22 }}
                  onClick={isUnlock ? handleUnlock : handleClear}
                  className="w-full flex items-center gap-4 bg-white/90 hover:bg-white rounded-2xl p-5 border border-sky-200 hover:border-amber-400/60 shadow-xl text-left"
                >
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.15 + i * 0.1, type: 'spring', stiffness: 300 }}
                    className="w-12 h-12 rounded-xl bg-amber-100 flex items-center justify-center shrink-0"
                  >
                    <Icon className="w-6 h-6 text-amber-600" />
                  </motion.div>
                  <div className="flex-1 min-w-0">
                    <span className="text-xl font-bold text-slate-800 block">{opt.name}</span>
                    <span className="text-base text-slate-500">
                      {isUnlock && 'Desbloquea todos los niveles del juego'}
                      {isClear && 'Elimina progreso y datos guardados'}
                    </span>
                  </div>
                  {done && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-emerald-600 text-base font-medium"
                    >
                      Listo
                    </motion.span>
                  )}
                </motion.button>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </motion.ul>
      </div>
    </div>
  );
}

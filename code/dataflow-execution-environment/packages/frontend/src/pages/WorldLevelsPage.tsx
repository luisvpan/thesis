import { Link, useNavigate, useParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowLeft } from 'lucide-react';
import { AnimatedMenuBackground } from '@/components/AnimatedMenuBackground';

const LEVELS = [1, 2, 3, 4] as const;
const WORLD_NAMES: Record<string, string> = {
  '1': 'Mundo 1',
  '2': 'Mundo 2',
  '3': 'Mundo 3',
  sandbox: 'Sandbox',
};

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.15 },
  },
};

const item = {
  hidden: { opacity: 0, scale: 0.9 },
  show: { opacity: 1, scale: 1 },
};

export default function WorldLevelsPage() {
  const { worldId } = useParams<{ worldId: string }>();
  const navigate = useNavigate();
  const worldName = worldId ? WORLD_NAMES[worldId] ?? `Mundo ${worldId}` : '';

  if (!worldId) {
    navigate('/juego');
    return null;
  }

  if (worldId === 'sandbox') {
    navigate('/ide/sandbox', { replace: true });
    return null;
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-sky-300 via-sky-200 to-emerald-100 flex flex-col items-center p-6 md:p-8">
      <AnimatedMenuBackground />
      <div className="relative z-10 w-full max-w-2xl flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8 w-full max-w-2xl"
        >
          <button
            type="button"
            onClick={() => navigate('/juego')}
            className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            Volver a Juego
          </button>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-4xl md:text-5xl font-black text-slate-800 mb-2 text-center"
        >
          {worldName}
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-slate-600 mb-10 text-center"
        >
          Elige un nivel para jugar
        </motion.p>

        <motion.ul
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-2 sm:grid-cols-4 gap-4 w-full max-w-md mx-auto justify-items-center"
        >
          <AnimatePresence>
            {LEVELS.map((level, i) => (
              <motion.li key={level} variants={item} className="w-full max-w-[140px]">
                <Link to={`/ide/${worldId}/${level}`} className="block">
                  <motion.div
                    whileHover={{ scale: 1.05, y: -4 }}
                    whileTap={{ scale: 0.98 }}
                    transition={{ type: 'spring', stiffness: 400, damping: 22 }}
                    className="aspect-square rounded-2xl bg-white/90 hover:bg-white border border-sky-200 hover:border-sky-400 flex flex-col items-center justify-center gap-2 shadow-xl"
                  >
                    <motion.span
                      initial={{ rotate: -10, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      transition={{ delay: 0.2 + i * 0.1 }}
                      className="text-4xl font-black text-slate-800"
                    >
                      {level}
                    </motion.span>
                    <span className="text-sm text-slate-500">Nivel</span>
                  </motion.div>
                </Link>
              </motion.li>
            ))}
          </AnimatePresence>
        </motion.ul>
      </div>
    </div>
  );
}

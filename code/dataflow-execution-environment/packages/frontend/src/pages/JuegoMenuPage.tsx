import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowLeft, Globe, FlaskConical } from 'lucide-react';
import { AnimatedMenuBackground } from '@/components/AnimatedMenuBackground';

const WORLDS = [
  { id: '1', name: 'Mundo 1', icon: Globe },
  { id: '2', name: 'Mundo 2', icon: Globe },
  { id: '3', name: 'Mundo 3', icon: Globe },
  { id: 'sandbox', name: 'Sandbox', icon: FlaskConical },
] as const;

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};

export default function JuegoMenuPage() {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-sky-300 via-sky-200 to-emerald-100 flex flex-col items-center p-6 md:p-8">
      <AnimatedMenuBackground />
      <div className="relative z-10 w-full max-w-4xl">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
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
          Juego
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-slate-600 mb-10 text-center"
        >
          Elige un mundo
        </motion.p>

        <motion.ul
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full mx-auto justify-items-center"
        >
          <AnimatePresence>
            {WORLDS.map((world, i) => {
              const Icon = world.icon;
              const isSandbox = world.id === 'sandbox';
              const to = isSandbox ? '/ide/sandbox' : `/juego/${world.id}`;
              return (
                <motion.li key={world.id} variants={item} className="w-full max-w-[220px]">
                  <Link to={to} className="block h-full">
                    <motion.div
                      whileHover={{ scale: 1.03, y: -6 }}
                      whileTap={{ scale: 0.98 }}
                      transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                      className="h-full bg-white/90 hover:bg-white rounded-2xl p-6 border border-sky-200 hover:border-sky-400 shadow-xl flex flex-col items-center gap-4 transition-colors"
                    >
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.2 + i * 0.08, type: 'spring', stiffness: 300 }}
                        className="w-14 h-14 rounded-xl bg-sky-100 flex items-center justify-center"
                      >
                        <Icon className="w-7 h-7 text-sky-600" />
                      </motion.div>
                      <span className="text-xl font-bold text-slate-800">{world.name}</span>
                      <span className="text-sm text-slate-500">
                        {isSandbox ? null : '4 niveles'}
                      </span>
                    </motion.div>
                  </Link>
                </motion.li>
              );
            })}
          </AnimatePresence>
        </motion.ul>
      </div>
    </div>
  );
}

import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { Gamepad2, Settings, ChevronRight, Terminal } from 'lucide-react';
import { AnimatedMenuBackground } from '@/components/AnimatedMenuBackground';
import { ApiHealthBadge } from '@/components/ApiHealthBadge';

const SECTIONS = [
  {
    id: 'juego',
    title: 'Juego',
    description: 'Mundos 1, 2 y 3 con niveles · Sandbox',
    to: '/juego',
    icon: Gamepad2,
    gradient: 'from-teal-500 to-cyan-600',
    border: 'border-teal-400/30',
    shadow: 'shadow-teal-500/25',
  },
  {
    id: 'configuracion',
    title: 'Configuración',
    description: 'Desbloquear niveles y borrar datos',
    to: '/configuracion',
    icon: Settings,
    gradient: 'from-amber-500 to-orange-600',
    border: 'border-amber-400/30',
    shadow: 'shadow-amber-500/25',
  },
  {
    id: 'desarrollador',
    title: 'Modo desarrollador',
    description: 'Diseña ejercicios dataflow y prueba el compilador sin visión',
    to: '/ide/dev',
    icon: Terminal,
    gradient: 'from-violet-500 to-indigo-600',
    border: 'border-violet-400/30',
    shadow: 'shadow-violet-500/25',
  },
] as const;

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.2 },
  },
};

const item = {
  hidden: { opacity: 0, y: 32 },
  show: { opacity: 1, y: 0 },
};

export default function HomePage() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-sky-300 via-sky-200 to-emerald-100 flex flex-col items-center justify-center p-6 md:p-8">
      <div className="fixed bottom-4 right-4 z-20">
        <ApiHealthBadge />
      </div>
      <AnimatedMenuBackground />
      <div className="relative z-10 flex flex-col items-center w-full">
      <motion.h1
        initial={{ opacity: 0, y: -24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="text-4xl md:text-6xl font-black text-slate-800 mb-2 text-center"
      >
        Menú principal
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
        className="text-slate-600 text-xl mb-14 text-center"
      >
        Elige una opción para continuar
      </motion.p>

      <motion.nav
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 w-full max-w-6xl"
      >
        <AnimatePresence>
          {SECTIONS.map((section, i) => {
            const Icon = section.icon;
            return (
              <motion.div key={section.id} variants={item} className="min-w-0">
                <Link to={section.to} className="block h-full">
                  <motion.div
                    whileHover={{ scale: 1.03, y: -8 }}
                    whileTap={{ scale: 0.98 }}
                    transition={{ type: 'spring', stiffness: 380, damping: 22 }}
                    className={`h-full bg-gradient-to-br ${section.gradient} rounded-2xl p-8 shadow-xl ${section.shadow} border ${section.border} flex flex-col`}
                  >
                    <motion.div
                      initial={{ rotate: -12, scale: 0.9 }}
                      animate={{ rotate: 0, scale: 1 }}
                      transition={{ delay: 0.3 + i * 0.15, type: 'spring', stiffness: 260 }}
                      className="w-16 h-16 rounded-2xl bg-white/20 flex items-center justify-center mb-4"
                    >
                      <Icon className="w-8 h-8 text-white" />
                    </motion.div>
                    <h2 className="text-3xl font-bold text-white mb-2">{section.title}</h2>
                    <p className="text-white/80 text-base flex-1 mb-4">{section.description}</p>
                    <motion.span
                      className="flex items-center gap-2 text-white font-medium text-base"
                      whileHover={{ x: 4 }}
                    >
                      Entrar
                      <ChevronRight className="w-4 h-4" />
                    </motion.span>
                  </motion.div>
                </Link>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </motion.nav>
      </div>
    </div>
  );
}

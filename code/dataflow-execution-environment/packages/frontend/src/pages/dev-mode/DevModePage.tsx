import { Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { ArrowLeft, FlaskConical } from 'lucide-react';
import { AnimatedMenuBackground } from '@/components/AnimatedMenuBackground';
import { useDevModePage } from './useDevModePage';

export default function DevModePage() {
  const { entries, goBack } = useDevModePage();

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-sky-300 via-sky-200 to-emerald-100 p-6 md:p-8">
      <AnimatedMenuBackground />
      <div className="relative z-10 mx-auto flex w-full max-w-4xl flex-col">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
          <button
            type="button"
            onClick={goBack}
            className="flex items-center gap-2 text-slate-600 transition-colors hover:text-slate-900"
          >
            <ArrowLeft className="h-5 w-5" />
            Volver al menú principal
          </button>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-2 text-center text-4xl font-black text-slate-800 md:text-5xl"
        >
          Modo Dev
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="mb-10 text-center text-slate-600"
        >
          Accesos rápidos para abrir el IDE con utilidades de desarrollo activadas
        </motion.p>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {entries.map((entry, index) => (
            <motion.div
              key={entry.id}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 * index }}
            >
              <Link
                to={entry.to}
                className="flex h-full flex-col rounded-2xl border border-violet-300/70 bg-white/90 p-5 shadow-xl transition-colors hover:bg-white"
              >
                <div className="mb-3 inline-flex h-11 w-11 items-center justify-center rounded-xl bg-violet-100">
                  <FlaskConical className="h-6 w-6 text-violet-600" />
                </div>
                <h2 className="mb-1 text-xl font-bold text-slate-800">{entry.title}</h2>
                <p className="text-sm text-slate-600">{entry.description}</p>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

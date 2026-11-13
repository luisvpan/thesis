import { Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { CircleHelp, Play, Volume2, X } from 'lucide-react';
import { MusicPlayerModal } from '@/components/MusicPlayerModal';
import { homeCubes, howToPlaySteps, useHomePage } from './useHomePage';

export default function HomePage() {
  const {
    containerMotion,
    itemMotion,
    showMusicModal,
    openMusicModal,
    closeMusicModal,
    showHowToPlay,
    openHowToPlay,
    closeHowToPlay,
  } = useHomePage();

  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-[#070b14] p-6 font-display">
      <div className="absolute right-6 top-6 flex gap-3">
        <button
          type="button"
          onClick={openMusicModal}
          className="flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-700 bg-slate-900 text-slate-200 transition-colors hover:bg-slate-800 hover:text-white"
          title="Música"
        >
          <Volume2 className="h-8 w-8 pointer-events-none" strokeWidth={2} />
        </button>
      </div>

      <motion.div
        variants={containerMotion}
        initial="hidden"
        animate="show"
        className="flex flex-col items-center"
      >
        <motion.div variants={itemMotion} className="mb-8 flex items-end gap-5">
          {homeCubes.map((cube, i) => (
            <motion.div
              key={cube.id}
              className={`rounded-2xl border-4 bg-slate-950 ${cube.size}`}
              style={{
                rotate: cube.rotate,
                borderColor: cube.color,
                boxShadow: `0 0 24px ${cube.color}80, inset 0 0 12px ${cube.color}33`,
              }}
              animate={{ y: [cube.y, cube.y - 10, cube.y] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: i * 0.4 }}
            />
          ))}
        </motion.div>

        <motion.h1
          variants={itemMotion}
          className="bg-gradient-to-r from-teal-300 via-sky-400 to-violet-400 bg-clip-text px-4 pb-3 text-center text-8xl font-bold tracking-tight text-transparent md:text-[10rem] md:leading-none"
        >
          WonderFlow
        </motion.h1>

        <motion.div
          variants={itemMotion}
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.97 }}
          transition={{ type: 'spring', stiffness: 380, damping: 22 }}
        >
          <Link
            to="/ide/sandbox"
            className="flex items-center gap-6 rounded-full bg-teal-400 py-5 pl-5 pr-16 text-slate-900 shadow-[0_0_48px_rgba(45,212,191,0.35)] ring-[10px] ring-teal-400/15 transition-colors hover:bg-teal-300"
          >
            <span className="flex h-24 w-24 items-center justify-center rounded-full bg-teal-950">
              <Play className="ml-1.5 h-11 w-11 fill-teal-400 text-teal-400" />
            </span>
            <span className="text-6xl font-bold md:text-7xl">Jugar</span>
          </Link>
        </motion.div>

        <motion.button
          variants={itemMotion}
          type="button"
          onClick={openHowToPlay}
          className="mt-10 flex items-center gap-3 rounded-full border border-slate-600 bg-slate-900/60 px-8 py-4 text-2xl font-semibold text-slate-100 transition-colors hover:bg-slate-800"
        >
          <CircleHelp className="h-8 w-8 pointer-events-none" strokeWidth={2} />
          ¿Cómo se juega?
        </motion.button>
      </motion.div>

      {showHowToPlay ? (
        <div className="fixed inset-0 z-200 flex items-center justify-center bg-black/70 p-6" onClick={closeHowToPlay}>
          <div
            role="dialog"
            aria-label="Cómo se juega"
            className="w-full max-w-2xl rounded-3xl border border-slate-700 bg-slate-900 p-8 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-6 flex items-center justify-between gap-4">
              <h2 className="text-4xl font-bold text-white">¿Cómo se juega?</h2>
              <button
                type="button"
                onClick={closeHowToPlay}
                className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border border-slate-700 bg-slate-800 text-slate-200 hover:bg-slate-700"
                title="Cerrar"
              >
                <X className="h-8 w-8 pointer-events-none" strokeWidth={2.5} />
              </button>
            </div>
            <ol className="flex flex-col gap-4">
              {howToPlaySteps.map((step, i) => (
                <li key={step} className="flex items-start gap-4 text-2xl text-slate-200">
                  <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-teal-400 font-bold text-slate-900">
                    {i + 1}
                  </span>
                  <span className="pt-1">{step}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      ) : null}

      {showMusicModal ? <MusicPlayerModal onClose={closeMusicModal} /> : null}
    </div>
  );
}

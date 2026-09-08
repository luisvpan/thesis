import { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { ArrowLeft, Maximize2, Minimize2 } from 'lucide-react';
import { CardGallery } from './components/cards/CardGallery';

interface CardGalleryPageProps {
  onBack?: () => void;
}

export default function CardGalleryPage({ onBack }: CardGalleryPageProps) {
  const [cardSize, setCardSize] = useState<'small' | 'medium' | 'large'>('medium');
  return (
    <div className="w-screen h-screen bg-white overflow-hidden relative">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-30 bg-gradient-to-r from-purple-500 to-pink-500 p-1 shadow-xl">
        <div className="bg-white px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {onBack ? (
              <motion.button
                onClick={onBack}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                className="bg-gradient-to-br from-gray-400 to-gray-600 rounded-full p-2 shadow-lg text-white"
              >
                <ArrowLeft className="w-5 h-5" />
              </motion.button>
            ) : (
              <Link to="/">
                <motion.span
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="inline-flex bg-gradient-to-br from-gray-400 to-gray-600 rounded-full p-2 shadow-lg text-white"
                >
                  <ArrowLeft className="w-5 h-5" />
                </motion.span>
              </Link>
            )}
            <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600">
              🎴 Galería de Cartas
            </h1>
          </div>

          {/* Controles de tamaño */}
          <div className="flex gap-2">
            <motion.button
              onClick={() => setCardSize('small')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`px-4 py-2 rounded-xl font-bold transition-all ${
                cardSize === 'small'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              <Minimize2 className="w-5 h-5" />
            </motion.button>
            <motion.button
              onClick={() => setCardSize('medium')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`px-4 py-2 rounded-xl font-bold transition-all ${
                cardSize === 'medium'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              M
            </motion.button>
            <motion.button
              onClick={() => setCardSize('large')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`px-4 py-2 rounded-xl font-bold transition-all ${
                cardSize === 'large'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              <Maximize2 className="w-5 h-5" />
            </motion.button>
          </div>
        </div>
      </div>

      {/* Gallery Content */}
      <div className="pt-20 h-full">
        <CardGallery cardSize={cardSize} />
      </div>
    </div>
  );
}

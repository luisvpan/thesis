import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { IdeLayout } from '@/layouts/IdeLayout';
import HomePage from './pages/home/HomePage';
import DataflowPage from './pages/dataflow/DataflowPage';
import { unlockSpeechAudio } from '@/utils/speakSpanish';
import { MusicPlayerProvider } from '@/contexts/MusicPlayerContext';

function App() {
  // Los toques que llegan por el WebSocket de la mesa táctil se simulan con
  // `element.click()`, que el navegador no cuenta como gesto de usuario. Un
  // solo evento real (mouse, teclado o toque físico en la pantalla) aquí
  // desbloquea el audio/voz para el resto de la sesión. Escuchamos todos los
  // eventos reales (no solo el primero) por si el navegador vuelve a
  // suspender el audio; unlockSpeechAudio() no hace nada si ya está activo.
  useEffect(() => {
    const unlock = (e: Event) => {
      if (e.isTrusted) unlockSpeechAudio();
    };
    window.addEventListener('pointerdown', unlock, { capture: true });
    window.addEventListener('keydown', unlock, { capture: true });
    return () => {
      window.removeEventListener('pointerdown', unlock, { capture: true });
      window.removeEventListener('keydown', unlock, { capture: true });
    };
  }, []);

  return (
    <MusicPlayerProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/ide" element={<IdeLayout />}>
            <Route path="sandbox" element={<DataflowPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </MusicPlayerProvider>
  );
}

export default App;

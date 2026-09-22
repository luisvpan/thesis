import { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { IdeLayout } from '@/layouts/IdeLayout';
import HomePage from './pages/home/HomePage';
import DataflowPage from './pages/dataflow/DataflowPage';
import JuegoMenuPage from './pages/juego-menu/JuegoMenuPage';
import WorldLevelsPage from './pages/world-levels/WorldLevelsPage';
import ConfiguracionPage from './pages/configuracion/ConfiguracionPage';
import DevModePage from './pages/dev-mode';
import { unlockSpeechAudio } from '@/utils/speakSpanish';
import { MusicPlayerProvider } from '@/contexts/MusicPlayerContext';

function App() {
  // Los toques que llegan por el WebSocket de la mesa táctil se simulan con
  // `element.click()`, que el navegador no cuenta como gesto de usuario. Un
  // solo evento real (mouse, teclado o toque físico en la pantalla) aquí
  // desbloquea el audio/voz para el resto de la sesión.
  useEffect(() => {
    const unlock = () => unlockSpeechAudio();
    window.addEventListener('pointerdown', unlock, { once: true, capture: true });
    window.addEventListener('keydown', unlock, { once: true, capture: true });
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
          <Route path="/juego" element={<JuegoMenuPage />} />
          <Route path="/juego/:worldId" element={<WorldLevelsPage />} />
          <Route path="/configuracion" element={<ConfiguracionPage />} />
          <Route path="/dev" element={<DevModePage />} />
          <Route path="/ide" element={<IdeLayout />}>
            <Route path="sandbox" element={<DataflowPage isSandbox={true} />} />
            <Route path=":worldId/:level" element={<DataflowPage isSandbox={false} />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </MusicPlayerProvider>
  );
}

export default App;

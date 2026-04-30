import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SocketProvider } from '@/contexts/SocketContext';
import { SocketNavigationListener } from '@/components/SocketNavigationListener';
import { IdeLayout } from '@/layouts/IdeLayout';
import HomePage from './pages/home/HomePage';
import DataflowPage from './pages/dataflow/DataflowPage';
import JuegoMenuPage from './pages/juego-menu/JuegoMenuPage';
import WorldLevelsPage from './pages/world-levels/WorldLevelsPage';
import ConfiguracionPage from './pages/configuracion/ConfiguracionPage';
import DevModePage from './pages/dev-mode';

function App() {
  return (
    <BrowserRouter>
      <SocketProvider>
        <SocketNavigationListener />
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
      </SocketProvider>
    </BrowserRouter>
  );
}

export default App;

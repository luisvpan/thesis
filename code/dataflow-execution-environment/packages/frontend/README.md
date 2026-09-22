# React + TypeScript + Vite

## Música de fondo

El reproductor arma su lista sola: `src/data/musicPlaylist.ts` hace
`import.meta.glob('/src/assets/*.{mp3,wav}')`, así que la lista es, literalmente, los
archivos de audio que haya en `src/assets/`. No hace falta registrar nada a mano.

**Los archivos de audio no están versionados**, a propósito: pesaban 180 MB —148 MB solo
en cinco WAV sin comprimir— y en git eso no se puede deshacer, porque los objetos quedan
en el historial aunque después se borren los archivos. Sin ellos el proyecto compila
igual y el reproductor aparece con la lista vacía.

Para reponerlos, deja los archivos en `src/assets/` teniendo en cuenta que:

- **MP3, no WAV.** Para música de fondo en un navegador la diferencia no se oye y el
  archivo pesa alrededor de una quinta parte.
- Si aun así el conjunto se va a decenas de MB, mejor no meterlo en el repo: o Git LFS, o
  servirlo desde fuera y apuntar `PLAYLIST` a esas URLs.
- El título de cada pista sale del nombre del archivo, sin extensión y sin el paréntesis
  final (`Aftertune - Crystals (Original Mix).mp3` → "Aftertune - Crystals").

## Plantilla

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is currently not compatible with SWC. See [this issue](https://github.com/vitejs/vite-plugin-react/issues/428) for tracking the progress.

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

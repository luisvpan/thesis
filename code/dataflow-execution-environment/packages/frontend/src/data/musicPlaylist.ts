export type Track = {
  title: string;
  url: string;
};

// Todos los archivos de audio en src/assets, resueltos por Vite a su URL final.
const modules = import.meta.glob('/src/assets/*.{mp3,wav}', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>;

function titleCase(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

/** Deriva un título limpio del nombre de archivo (sin extensión ni anotaciones de mezcla/versión). */
function titleFromPath(path: string): string {
  const filename = path.split('/').pop() ?? path;
  const withoutExt = filename.replace(/\.(mp3|wav)$/i, '');
  const withoutParens = withoutExt.replace(/\s*\([^)]*\)\s*$/, '').trim();
  if (withoutParens.includes(' ')) return withoutParens;
  return withoutParens
    .replace(/[-_]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map(titleCase)
    .join(' ');
}

export const PLAYLIST: Track[] = Object.entries(modules)
  .map(([path, url]) => ({ title: titleFromPath(path), url }))
  .sort((a, b) => a.title.localeCompare(b.title));

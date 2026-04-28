import {
  DECK_SECTION_ITEMS,
  deckLabel,
  type DeckSectionId,
} from '@/data/yoloDeckCatalog';
import { useNode } from '@/contexts/NodeContext';

const SECTION_TITLE: Record<DeckSectionId, string> = {
  numbers: 'Números',
  operators: 'Operadores',
  figures: 'Figuras',
  foods: 'Comidas',
};

/** Acento de borde/fondo por categoría (números azul, operadores rojo, figuras amarillo, comidas naranja). */
const SECTION_PANEL: Record<DeckSectionId, string> = {
  numbers: 'border-blue-500/50 bg-blue-950/25',
  operators: 'border-red-500/50 bg-red-950/25',
  figures: 'border-yellow-400/50 bg-yellow-950/20',
  foods: 'border-orange-500/50 bg-orange-950/25',
};

const SECTION_BUTTON: Record<DeckSectionId, string> = {
  numbers:
    'border-blue-500/80 bg-slate-800 hover:bg-blue-900/60 text-blue-100',
  operators:
    'border-red-500/80 bg-slate-800 hover:bg-red-900/50 text-red-100',
  figures:
    'border-yellow-500/80 bg-slate-800 hover:bg-yellow-900/40 text-yellow-100',
  foods:
    'border-orange-500/80 bg-slate-800 hover:bg-orange-900/40 text-orange-100',
};

const ORDER: DeckSectionId[] = ['numbers', 'operators', 'figures', 'foods'];

export function ModelDeckSidebar() {
  const { spawnDeckYoloClass } = useNode();

  return (
    <div className="space-y-4">
      {ORDER.map((section) => (
        <div
          key={section}
          className={`rounded-xl border p-3 ${SECTION_PANEL[section]}`}
        >
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            {SECTION_TITLE[section]}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {DECK_SECTION_ITEMS[section].map((yolo) => (
              <button
                key={yolo}
                type="button"
                title={yolo}
                onClick={() => spawnDeckYoloClass(yolo)}
                className={`min-h-[2.25rem] rounded-lg border px-2 py-1.5 text-sm font-semibold transition-colors ${SECTION_BUTTON[section]}`}
              >
                {deckLabel(yolo)}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

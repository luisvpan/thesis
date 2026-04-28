import { DECK_SECTION_ITEMS, deckLabel, type DeckSectionId } from '@/data/yoloDeckCatalog';
import { useNode } from '@/contexts/NodeContext';

const TITLES: Record<DeckSectionId, string> = {
  numbers: 'Numeros',
  operators: 'Operadores',
  figures: 'Figuras',
  foods: 'Comidas',
};

const COLORS: Record<DeckSectionId, string> = {
  numbers: 'border-blue-500 text-blue-100',
  operators: 'border-red-500 text-red-100',
  figures: 'border-yellow-500 text-yellow-100',
  foods: 'border-orange-500 text-orange-100',
};

export function ModelDeckSidebar() {
  const { spawnDeckYoloClass } = useNode();
  const order: DeckSectionId[] = ['numbers', 'operators', 'figures', 'foods'];

  return (
    <div className="space-y-3">
      {order.map((section) => (
        <section key={section}>
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">{TITLES[section]}</p>
          <div className="flex flex-wrap gap-1">
            {DECK_SECTION_ITEMS[section].map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => spawnDeckYoloClass(c)}
                className={`px-2 py-1 rounded border bg-slate-800 hover:bg-slate-700 text-xs font-semibold ${COLORS[section]}`}
              >
                {deckLabel(c)}
              </button>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

import {
  CPA_YOLO_SIDEBAR_SECTIONS,
  deckLabel,
  type CpaSidebarSectionId,
} from '@/data/yoloDeckCatalog';
import { useNode } from '@/contexts/NodeContext';

const SECTION_STYLES: Record<CpaSidebarSectionId, string> = {
  concreto: 'border-orange-500 text-orange-100',
  pictorico: 'border-yellow-500 text-yellow-100',
  abstracto: 'border-blue-500 text-blue-100',
  comun: 'border-teal-500 text-teal-100',
};

export function ModelDeckSidebar() {
  const { spawnDeckYoloClass } = useNode();

  return (
    <div className="space-y-4">
      {CPA_YOLO_SIDEBAR_SECTIONS.map((section) => (
        <section key={section.id}>
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">{section.title}</p>
          <div className="flex flex-wrap gap-1">
            {section.yoloClasses.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => spawnDeckYoloClass(c)}
                className={`px-2 py-1 rounded border bg-slate-800 hover:bg-slate-700 text-xs font-semibold ${SECTION_STYLES[section.id]}`}
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

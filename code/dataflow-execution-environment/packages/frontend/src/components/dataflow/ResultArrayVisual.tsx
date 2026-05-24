import type { ResultVisualItem } from '@/services/executeProgram';
import {
  MontessoriCubeGlyph,
  FormaGlyph,
  ComidaGlyph,
  CapGlyph,
  StickGlyph,
} from './CpaGlyphs';

type ResultArrayVisualProps = {
  items: ResultVisualItem[];
  /** Optional grouping info for multiplication results */
  grouping?: { groupSize: number; groupCount: number };
  /** Alignment of items: 'start' or 'center' (default) */
  align?: 'start' | 'center';
};

const MAX_SHOW = 36;

function renderGlyph(item: ResultVisualItem, index: number) {
  const key = `v-${index}`;
  switch (item.kind) {
    case 'montessori':
      return <MontessoriCubeGlyph key={key} color={item.color} />;
    case 'forma':
      return <FormaGlyph key={key} subtype={item.subtype} color={item.color} />;
    case 'cap':
      return <CapGlyph key={key} color={item.color} />;
    case 'stick':
      return <StickGlyph key={key} color={item.color} />;
    case 'comida':
      return <ComidaGlyph key={key} subtype={item.subtype} color={item.color} />;
    default:
      return null;
  }
}

export function ResultArrayVisual({ items, grouping, align = 'center' }: ResultArrayVisualProps) {
  if (items.length === 0) return null;

  const shown = items.slice(0, MAX_SHOW);
  const overflow = items.length - shown.length;
  const itemsAlign = align === 'start' ? 'items-start' : 'items-center';
  const flexJustify = align === 'start' ? 'justify-start' : 'justify-center';

  // Without grouping: flat render
  if (!grouping) {
    return (
      <div className={`flex w-full max-w-44 flex-col gap-1 ${itemsAlign}`}>
        <div className={`flex flex-wrap gap-1.5 ${flexJustify}`}>
          {shown.map((item, i) => renderGlyph(item, i))}
        </div>
        {overflow > 0 ? (
          <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
        ) : null}
      </div>
    );
  }

  // With grouping: split into visual groups
  const { groupSize, groupCount } = grouping;
  const groups: React.ReactNode[] = [];

  for (let g = 0; g < groupCount; g++) {
    const start = g * groupSize;
    const end = Math.min(start + groupSize, shown.length);
    if (start >= shown.length) break;

    groups.push(
      <div
        key={`group-${g}`}
        className="flex flex-wrap justify-center gap-1 p-1.5 rounded-md bg-slate-700/40 ring-1 ring-slate-600/50"
      >
        {shown.slice(start, end).map((item, i) => renderGlyph(item, start + i))}
      </div>
    );
  }

  return (
    <div className={`mt-2 flex w-full max-w-52 flex-col gap-1 ${itemsAlign}`}>
      <div className={`flex flex-wrap gap-4 ${flexJustify}`}>
        {groups}
      </div>
      {overflow > 0 ? (
        <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
      ) : null}
    </div>
  );
}

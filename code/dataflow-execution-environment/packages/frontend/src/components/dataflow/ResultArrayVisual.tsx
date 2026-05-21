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
  align?: 'start' | 'center';
};

const MAX_SHOW = 36;

export function ResultArrayVisual({ items, align = 'center' }: ResultArrayVisualProps) {
  if (items.length === 0) return null;

  const shown = items.slice(0, MAX_SHOW);
  const overflow = items.length - shown.length;
  const itemsAlign = align === 'start' ? 'items-start' : 'items-center';
  const flexJustify = align === 'start' ? 'justify-start' : 'justify-center';

  return (
    <div className={`flex w-full max-w-44 flex-col gap-1 ${itemsAlign}`}>
      <div className={`flex flex-wrap gap-1.5 ${flexJustify}`}>
        {shown.map((item, i) => {
          if (item.kind === 'montessori') {
            return <MontessoriCubeGlyph key={`v-${i}`} color={item.color} />;
          }
          if (item.kind === 'forma') {
            return (
              <FormaGlyph
                key={`v-${i}`}
                subtype={item.subtype}
                color={item.color}
              />
            );
          }
          if (item.kind === 'cap') {
            return <CapGlyph key={`v-${i}`} color={item.color} />;
          }
          if (item.kind === 'stick') {
            return <StickGlyph key={`v-${i}`} color={item.color} />;
          }
          return (
            <ComidaGlyph
              key={`v-${i}`}
              subtype={item.subtype}
              color={item.color}
            />
          );
        })}
      </div>
      {overflow > 0 ? (
        <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
      ) : null}
    </div>
  );
}

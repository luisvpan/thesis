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
};

const MAX_SHOW = 36;

export function ResultArrayVisual({ items }: ResultArrayVisualProps) {
  if (items.length === 0) return null;

  const shown = items.slice(0, MAX_SHOW);
  const overflow = items.length - shown.length;

  return (
    <div className="mt-2 flex w-full max-w-44 flex-col items-center gap-1">
      <div className="flex flex-wrap justify-center gap-1.5">
        {shown.map((item, i) => {
          if (item.kind === 'montessori') {
            return <MontessoriCubeGlyph key={`v-${i}`} color={item.color} />;
          }
          if (item.kind === 'forma') {
            return <FormaGlyph key={`v-${i}`} subtype={item.subtype} />;
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

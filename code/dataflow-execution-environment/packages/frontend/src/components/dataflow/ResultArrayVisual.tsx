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
  /** Alignment of items: 'start' or 'center' (default) */
  align?: 'start' | 'center';
  /** Tamaño ampliado para la carta de resultado; por defecto, tamaño del token que viaja por las conexiones. */
  large?: boolean;
};

const MAX_SHOW = 36;

function renderGlyph(item: ResultVisualItem, index: number, large: boolean) {
  const key = `v-${index}`;
  switch (item.kind) {
    case 'montessori':
      return <MontessoriCubeGlyph key={key} color={item.color} large={large} />;
    case 'forma':
      return (
        <FormaGlyph
          key={key}
          subtype={item.subtype}
          size={item.size}
          color={item.color}
          large={large}
        />
      );
    case 'cap':
      return <CapGlyph key={key} color={item.color} large={large} />;
    case 'stick':
      return <StickGlyph key={key} color={item.color} large={large} />;
    case 'comida':
      return <ComidaGlyph key={key} subtype={item.subtype} color={item.color} large={large} />;
    default:
      return null;
  }
}

export function ResultArrayVisual({
  items,
  align = 'center',
  large = false,
}: ResultArrayVisualProps) {
  if (items.length === 0) return null;

  const shown = items.slice(0, MAX_SHOW);
  const overflow = items.length - shown.length;
  const itemsAlign = align === 'start' ? 'items-start' : 'items-center';
  const flexJustify = align === 'start' ? 'justify-start' : 'justify-center';
  const maxWClass = large ? 'max-w-64' : 'max-w-44';
  const gapClass = large ? 'gap-2.5' : 'gap-1.5';

  return (
    <div className={`flex w-full ${maxWClass} flex-col gap-1 ${itemsAlign}`}>
      <div className={`flex flex-wrap ${gapClass} ${flexJustify}`}>
        {shown.map((item, i) => renderGlyph(item, i, large))}
      </div>
      {overflow > 0 ? (
        <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
      ) : null}
    </div>
  );
}

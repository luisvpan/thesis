/** Track ID de ByteTrack — solo badge, sin alterar la carta. */
export function TrackIdBadge({ trackId }: { trackId?: number }) {
  if (trackId === undefined || trackId < 0) return null;

  return (
    <span className="pointer-events-none absolute right-1 top-1 z-10 rounded bg-indigo-600 px-1.5 py-0.5 font-mono text-[10px] font-bold text-white shadow-sm">
      #{trackId}
    </span>
  );
}

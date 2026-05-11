type Props = { show: boolean };

export function CpaModeDisabledBanner({ show }: Props) {
  if (!show) return null;
  return (
    <div
      role="alert"
      className="mb-1.5 max-w-[13rem] rounded-md border border-red-600/90 bg-red-950/95 px-2 py-1 text-center text-[10px] font-semibold leading-tight text-red-50 shadow-lg"
    >
      No se puede usar en el modo actual.
    </div>
  );
}

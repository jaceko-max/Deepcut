export function Warnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <ul className="space-y-1 rounded-lg border border-line bg-surface-2 p-3 text-sm" aria-label="Warnings">
      {warnings.map((w) => (
        <li key={w}>Heads up: {w}</li>
      ))}
    </ul>
  );
}

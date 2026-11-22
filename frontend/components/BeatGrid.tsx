// Decorative 16-step beat grid. Heights are a fixed kick/snare/hat pattern, not data.
const STEPS = [1, 0.25, 0.45, 0.25, 0.8, 0.25, 0.45, 0.3, 1, 0.25, 0.6, 0.25, 0.8, 0.3, 0.45, 0.55];

export function BeatGrid({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden className={`flex h-16 items-end gap-1 sm:gap-1.5 ${className}`}>
      {STEPS.map((h, i) => (
        <span
          key={i}
          className="wave-bar block flex-1 rounded-[2px] bg-accent"
          style={{ height: "100%", ["--rest" as string]: h, ["--i" as string]: i, opacity: i % 4 === 0 ? 1 : 0.55 }}
        />
      ))}
    </div>
  );
}

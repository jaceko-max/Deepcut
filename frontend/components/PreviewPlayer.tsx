"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { previewSrc } from "@/lib/api";
import type { Result } from "@/lib/types";

interface PreviewCtx {
  playingId: number | null;
  failedId: number | null;
  toggle: (r: Result) => void;
}

const Ctx = createContext<PreviewCtx | null>(null);

/** Owns the single shared <audio>; only one preview plays at a time. */
export function PreviewProvider({ children }: { children: React.ReactNode }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const [failedId, setFailedId] = useState<number | null>(null);
  const currentRef = useRef<number | null>(null);
  const directRef = useRef<string | null>(null);

  // The proxy link can fail if the free server restarted; Deezer's signed URL stays valid for ~15 minutes.
  const fail = useCallback(() => {
    const a = audioRef.current;
    const direct = directRef.current;
    if (a && direct && a.src !== direct) {
      directRef.current = null;
      a.src = direct;
      a.play().catch(() => {});
      return;
    }
    setFailedId(currentRef.current);
    setPlayingId(null);
  }, []);

  const toggle = useCallback((r: Result) => {
    const a = audioRef.current;
    if (!a) return;
    setFailedId(null);
    if (currentRef.current === r.deezer_id && !a.paused) {
      a.pause();
      return;
    }
    if (currentRef.current !== r.deezer_id) {
      currentRef.current = r.deezer_id;
      const src = previewSrc(r);
      directRef.current = r.preview_url && r.preview_url !== src ? r.preview_url : null;
      a.src = src;
    }
    a.play().catch((e: unknown) => {
      if (e instanceof DOMException && e.name === "NotAllowedError") {
        setFailedId(r.deezer_id);
        setPlayingId(null);
      }
      // Load failures also fire the <audio> error event, which runs the fallback.
    });
  }, []);

  useEffect(() => {
    const a = audioRef.current;
    return () => a?.pause();
  }, []);

  const value = useMemo(() => ({ playingId, failedId, toggle }), [playingId, failedId, toggle]);

  return (
    <Ctx.Provider value={value}>
      {children}
      <audio
        ref={audioRef}
        preload="none"
        onPlay={() => setPlayingId(currentRef.current)}
        onPause={() => setPlayingId(null)}
        onEnded={() => setPlayingId(null)}
        onError={fail}
      />
    </Ctx.Provider>
  );
}

export function usePreview() {
  const c = useContext(Ctx);
  if (!c) throw new Error("usePreview must be used inside PreviewProvider");
  return c;
}

export function PlayButton({ result }: { result: Result }) {
  const { playingId, failedId, toggle } = usePreview();
  const playing = playingId === result.deezer_id;
  const failed = failedId === result.deezer_id;
  return (
    <div className="flex flex-col items-start gap-1">
      <button
        type="button"
        onClick={() => toggle(result)}
        aria-pressed={playing}
        aria-label={`${playing ? "Pause" : "Play"} preview of ${result.title} by ${result.artist}`}
        className="grid size-12 shrink-0 place-items-center rounded-full bg-accent text-accent-ink transition hover:brightness-110"
      >
        {playing ? (
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="currentColor"><rect x="2" y="1" width="4" height="14" rx="1" /><rect x="10" y="1" width="4" height="14" rx="1" /></svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="currentColor"><path d="M3 1.5v13l11-6.5z" /></svg>
        )}
      </button>
      {failed && <span role="status" className="text-xs text-danger">Preview unavailable</span>}
    </div>
  );
}

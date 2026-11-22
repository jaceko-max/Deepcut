"use client";

import { useEffect, useState } from "react";

type Theme = "dark" | "light";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    let t: Theme = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    try {
      const saved = localStorage.getItem("deepcut-theme");
      if (saved === "light" || saved === "dark") t = saved;
    } catch {}
    document.documentElement.dataset.theme = t;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTheme(t);
  }, []);

  function toggle() {
    const next: Theme = theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    setTheme(next);
    try { localStorage.setItem("deepcut-theme", next); } catch {}
  }

  return (
    <button
      type="button"
      onClick={toggle}
      className="min-h-11 rounded-full border border-line px-4 text-sm text-muted hover:text-fg"
      aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
    >
      {theme === "light" ? "Dark" : "Light"}
    </button>
  );
}

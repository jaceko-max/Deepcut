export const SLIDER_MIN = 0;
export const SLIDER_MAX = 100;
/** Default listener cap. */
export const DEFAULT_LISTENERS = 500_000;

/** value = round(10^(4 + 2t/100)) : t=0 -> 10,000 ; t=100 -> 1,000,000 */
export function sliderToListeners(t: number): number {
  const c = Math.min(SLIDER_MAX, Math.max(SLIDER_MIN, t));
  return Math.round(Math.pow(10, 4 + (2 * c) / 100));
}

/** Exact (unrounded) slider position for a listener count. */
function exactSlider(value: number): number {
  return (100 * (Math.log10(value) - 4)) / 2;
}

export function listenersToSlider(value: number): number {
  return Math.min(SLIDER_MAX, Math.max(SLIDER_MIN, Math.round(exactSlider(value))));
}

/**
 * t≈84.95 maps to exactly 500,000 (the integer t=85 would read "501K"). The range input renders the thumb
 * at the nearest step; once the user drags, values are whole steps again.
 */
export const SLIDER_DEFAULT = exactSlider(DEFAULT_LISTENERS);

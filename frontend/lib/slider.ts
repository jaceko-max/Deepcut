export const SLIDER_MIN = 0;
export const SLIDER_MAX = 100;
/** t=65 maps to ~200K listeners. */
export const SLIDER_DEFAULT = 65;

/** value = round(10^(4 + 2t/100)) : t=0 -> 10,000 ; t=100 -> 1,000,000 */
export function sliderToListeners(t: number): number {
  const c = Math.min(SLIDER_MAX, Math.max(SLIDER_MIN, t));
  return Math.round(Math.pow(10, 4 + (2 * c) / 100));
}

export function listenersToSlider(value: number): number {
  const t = (100 * (Math.log10(value) - 4)) / 2;
  return Math.min(SLIDER_MAX, Math.max(SLIDER_MIN, Math.round(t)));
}

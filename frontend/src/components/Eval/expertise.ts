import type { RouteVariant } from '../../services/evalApi'

export interface LevelMeta {
  key: string
  label: string
  color: string
  /** Roughly-equivalent SAC hiking scale, for the legend. */
  sac: string
}

// Expertise levels, easiest → most committing (mirrors the backend's
// EXPERTISE_LEVELS order). Colors are deliberately distinct from the polylines
// already on the map — optimal (#2563eb blue), drawn (#f97316 orange) and hover
// (#dc2626 red) — so a route family reads clearly alongside them.
export const EXPERTISE_ORDER = ['casual', 'hiker', 'scrambler', 'alpinist'] as const

export const LEVEL_META: Record<string, LevelMeta> = {
  casual: { key: 'casual', label: 'Casual', color: '#16a34a', sac: 'T1' },
  hiker: { key: 'hiker', label: 'Hiker', color: '#0d9488', sac: 'T2' },
  scrambler: { key: 'scrambler', label: 'Scrambler', color: '#9333ea', sac: 'T3–T4' },
  alpinist: { key: 'alpinist', label: 'Alpinist', color: '#db2777', sac: 'T5–T6' },
}

export function levelColor(level: string): string {
  return LEVEL_META[level]?.color ?? '#64748b'
}

export function levelLabel(level: string): string {
  return LEVEL_META[level]?.label ?? level
}

export interface VariantDisplay {
  variant: RouteVariant
  color: string
  hasRoute: boolean
  duplicateOf?: string
  /** Draw a polyline for this variant: it is visible, has a route, and is not a
   *  duplicate of an already-drawn level (the backend marks identical lines with
   *  `duplicateOf` so we render each distinct route once). */
  draw: boolean
}

/** Decide, per variant, whether/how to render it — preserving the given order.
 *  `isVisible(level)` gates drawing so the user can toggle levels on and off. */
export function variantsForDisplay(
  variants: RouteVariant[],
  isVisible: (level: string) => boolean,
): VariantDisplay[] {
  return variants.map((variant) => {
    const hasRoute = variant.path.length > 0
    const duplicateOf = variant.duplicateOf
    return {
      variant,
      color: levelColor(variant.level),
      hasRoute,
      duplicateOf,
      draw: hasRoute && !duplicateOf && isVisible(variant.level),
    }
  })
}

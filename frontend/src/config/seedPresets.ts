import type { SeedLocation } from '../hooks/useSavedLocations'

/**
 * Optional, git-ignored preset seed. Create `presets.local.json` (see
 * `presets.local.example.json`) to seed your personal presets — e.g. home — on
 * first load. The glob resolves to `{}` when the file is absent, so the build
 * works with or without it, and the real file (with a real address) never enters
 * git.
 */
const modules = import.meta.glob('./presets.local.json', { eager: true }) as Record<
  string,
  { default: SeedLocation[] }
>

export const seedPresets: SeedLocation[] = Object.values(modules)[0]?.default ?? []

import type { SeedLocation } from '../hooks/useSavedLocations'

/**
 * Optional, git-ignored preset seed. Create `presets.local.json` (see
 * `presets.local.example.json`) to seed your personal presets — e.g. home — on
 * first load. The glob resolves to `{}` when the file is absent, so the build
 * works with or without it, and the real file (with a real address) never enters
 * git.
 *
 * CAVEAT: this seed is imported at build time, so if `presets.local.json` exists
 * when you run a *production* build (`vite build`), its contents are baked into
 * the client JS bundle. Git is covered, but a public deployment built with the
 * file present would ship your address. For a localhost-only dev app this is fine;
 * if you ever deploy publicly, build without the file (its data is already in each
 * user's localStorage from first run).
 */
const modules = import.meta.glob('./presets.local.json', { eager: true }) as Record<
  string,
  { default: SeedLocation[] }
>

export const seedPresets: SeedLocation[] = Object.values(modules)[0]?.default ?? []

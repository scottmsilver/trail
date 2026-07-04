# Terrain Passability: Extent-Aware Slope Gating & Expertise Levels

**Status:** findings + design proposal (corpus-validated; not yet implemented)
**Date:** 2026-07-04
**Context:** A real walked GPX route (Wasatch Crest via Dream Peak) scored *impassable*
while the engine's optimal to the same summit scored passable. Investigating that led to
a corpus study over the Wasatch and a reframing of passability as an **expertise level**.

---

## 1. The problem

The engine is a **terrain router**: it finds routes over the ground itself, staying on
trails where they exist but free to leave them. Passability is judged from a DEM, not a
trail map.

Today's impassability test is **memoryless**: one grid cell-to-cell move steeper than
`max_slope_degrees` (default **45°**) returns `inf`. A single steep cell kills the whole
path. This produced a **false positive** on a genuinely-walked route — and, as the corpus
study below shows, it does so on a large fraction of the *real trail network*.

## 2. Evidence at the source (Dream Peak)

Three elevation sources at the flagged spots, plus the hiker's own GPS:

| Source | Res | Reading |
|---|---|---|
| Engine DEM (USGS 3DEP 2020) | 10 m | ~50° → impassable |
| USGS 3DEP (py3dep) | 1 m | 58–73° raw, ~30° windowed |
| **USFS 2023 LiDAR** (raw point cloud) | 0.5 m | **real** ~11 m steep bank; 51° over 5 m |
| Hiker's GPS track | — | **3–6°** (threaded a gentle line between features) |

The terrain is genuinely rough — real steep banks with **gentle corridors weaving between
them**. Coarse 10 m data *smears* bank + corridor into one false 50° wall. The failure is
two-sided: **coarse data invents walls**, and **any-resolution point-slope false-blocks a
committed line** that clips a cut-bank (GPS error or a 1 m trail step).

## 3. Root cause

`slope > 45° ⇒ impassable` is **memoryless and scale-blind.** It cannot tell a **2–4 m
scramble-able step** from a **10 m+ cliff**, nor a **real step** from **coarse-DEM
smearing / GPS clip noise**. Both distinctions need (a) the steep run's **vertical extent**
and (b) enough **resolution** to see it.

## 4. The fix: extent-aware passability on fine data

Replace the memoryless gate with a **stateful, extent-aware** one.

### 4.1 Metric — "worst continuous steep climb"
Travelling in the direction of motion, accumulate a **steep debt** = vertical meters gained
while slope stays above the angle threshold; **reset to zero** at any bench/flat. The
move/path is impassable iff the steep debt exceeds a **budget**.

```
threshold_deg = 45           # what counts as "steep"
budget_m      = per expertise level (see §6)

debt = 0
for each step along the move (fine resolution):
    if slope(step) > threshold_deg:  debt += vertical_rise(step)
    else:                            debt  = 0
    if debt > budget_m:              return IMPASSABLE
return PASSABLE               # optionally: scramble cost penalty ∝ debt
```

### 4.2 Adaptive resolution
1 m everywhere is ~100× the cells of 10 m. Route on **10 m**; when a move trips the steep
gate, pull **1 m** for that patch and re-evaluate the extent there (coarse data can't
separate a real 3 m step from a smear). Refine only where it matters.

## 5. Corpus validation (the new work)

Pulled the real trail network from OpenStreetMap (overpass.openstreetmap.fr mirror):
**94 named Wasatch/PC peaks, 2054 trail ways.** Scored trail geometry **terrain-only**
(trail affinity off — being on a trail gives *zero* discount, so we test the ground, not
the label) against **17 USGS 1 m LiDAR tiles** downloaded for the region (2.5 GB, sampled
locally). Every mapped trail is **passable by definition** — so any "impassable" is a
false positive to be counted.

### 5.1 Ordinary trails — extent-aware is essentially perfect
248 trails fully inside the 1 m coverage:

| Gate (1 m) | False-blocks |
|---|---|
| Point-slope (today's engine) | **6%** |
| Extent-aware (4 m budget) | **1%** (rescues 14 of 16) |

Worst continuous climb: median 0 m, 95th percentile 1.5 m. And the false-block rate
**tracks OSM's own difficulty rating** — the tell that the metric is measuring something
real:

| `sac_scale` | n | point-slope | extent-aware |
|---|---|---|---|
| untagged | 139 | 1% | **0%** |
| hiking | 38 | 0% | **0%** |
| mountain_hiking | 60 | 8% | **0%** |
| demanding_mountain_hiking | 7 | 71% | 14% |
| alpine_hiking | 3 | 100% | 33% |

On the 237 ordinary trails people actually walk, extent-aware false-blocks **0%** while
point-slope wrongly kills up to 8%.

### 5.2 Demanding/alpine — the residual flags are *correct*
Pulled 72 hard-rated scrambles across the greater Wasatch; scored 35 within coverage:

| Gate | Flagged |
|---|---|
| Point-slope | 63% |
| Extent-aware (4 m) | 14% |

Every trail extent-aware flags is a **genuine class-3 scramble**, not a false positive:

| Trail | SAC | Worst continuous climb |
|---|---|---|
| **Mount Olympus Trail** | alpine_hiking | **11.0 m** |
| Everest Ridge | alpine_hiking | 11.1 m |
| Relsek Trail | demanding_mountain_hiking | 12.3 m |
| (unnamed) | alpine_hiking | 6.4 m |
| (unnamed) | alpine_hiking | 6.2 m |

Mount Olympus's summit push *is* an 11 m hands-on scramble. Extent-aware isn't wrongly
blocking these — it's **correctly detecting they aren't walks.**

## 6. The reframe: passability is an expertise level

The worst-continuous-climb is not a binary gate — it's a **physical difficulty signal.**
On a graded trail it's ~0 m; on Mount Olympus it's 11 m. It *is* the scramble height. So
the budget becomes a **per-expertise-level parameter**, calibrated from the corpus and
anchored to the SAC scale:

| Expertise level | Climb budget | ≈ SAC grade | Passes | Corpus behavior |
|---|---|---|---|---|
| **Casual** | ~1.5 m | T1 hiking | graded/valley trails | ~0% blocked |
| **Hiker** (default) | **~4 m** | T2 mountain_hiking | all ordinary mountain trails | **0%** false-block below alpine |
| **Scrambler** | ~8 m | T3–T4 demanding/alpine | most alpine routes | flags only the hardest |
| **Alpinist** | ~12–15 m (or penalty-only) | T5–T6 | class-3 pitches (Olympus) | passes everything |

The budget sweep on the demanding set confirms the tiers: 4 m → 14% flagged, 8 m → 9%,
12 m → 3%, 15 m → 0%. A casual hiker's router should *refuse* Mount Olympus's summit
gully; an alpinist's should allow it (perhaps with a steep cost penalty). **Same terrain,
different expertise, different verdict** — which point-slope could never express.

This plugs directly into the engine's existing user profiles (`easy` / `experienced` /
`trail_runner` / `accessibility`): each gains a `scrambleBudgetM`. The eval tool surfaces
worst-continuous-climb per route so the budgets stay calibrated against known-good tracks.

## 7. What we are NOT doing, and why
- **Wide slope windows / DEM smoothing** — a ~30 m (90 ft) window blurs away real short
  features along with noise → *misses* obstacles. Rejected.
- **Trusting the trail map** — the router must go off-trail; a mapped trail is a strong
  signal that can *relax* the budget on-trail, not the primary gate.
- **Buying commercial elevation data** — for the US mountains, free USGS/USFS LiDAR
  (0.5–1 m, ground-measured) beats every global commercial product, and in forest the
  satellite/radar DEMs see canopy, not ground. Commercial only helps coverage gaps abroad.

## 8. Honest limits
- **Centerlines are a proxy** — the corpus scores mapped centerlines, not recorded tracks;
  the direction is robust, exact percentages are soft.
- **Corridor narrower than the finest data → still invisible.**
- **DEM holds shape, not conditions** — a ≤45° ledge can still be scree, ice, or exposure.
  Extent-aware gives "geometrically negotiable at expertise level X," not "safe today."
- **Coverage** — N is limited to trails inside the downloaded 1 m tiles.

## 9. Data sources (reference)
- **Engine DEM:** USGS 3DEP via `py3dep.get_dem(bounds, resolution=…)` — 1 m free.
- **Bulk 1 m DEM tiles (fast, no per-request throttling):** USGS TNM S3
  (`prd-tnm` / `rockyweb.usgs.gov`), 10 km GeoTIFF tiles, found via the TNM products API by
  bbox. This is how the corpus was scored (17 tiles, EPSG:26912).
- **USFS 2023 LiDAR point cloud:** `rockyweb.usgs.gov/.../LPC/Projects/NV_USFSR4_D23/…`.
- **Trails + peaks:** OSM via `overpass.openstreetmap.fr` mirror (`overpass-api.de` is
  firewalled from our host). Recorded tracks: OSM GPS traces API.
- Analysis tooling added to `trail_env`: `laspy[lazrs]`, `matplotlib`, `markdown`, `rasterio`.

## 10. Implementation plan (TDD)
1. **Extent-aware metric in the scorer** — worst-continuous-climb per scored line; expose
   it in the eval API. Re-score the Dream Peak GPX → passable at 4 m; a synthetic cliff
   still blocks. Tests + security pass.
2. **Expertise-level budgets** — add `scrambleBudgetM` to the user profiles; the gate reads
   it. Casual refuses Olympus; alpinist allows it.
3. **Extent-aware gate in the core pathfinder** (+ native A* kernel), guarding golden/perf.
4. **Adaptive 1 m refinement** on a coarse steep-flag (respect the data-loading policy).
5. **Calibration surface** — eval reports worst-continuous-climb vs SAC so budgets stay tuned.

## 11. Open questions
- Should over-budget terrain be a hard block or a steep **cost penalty** that grows with
  climb height (so the router uses a scramble only when there's no alternative)?
- On-trail budget relaxation — a graded switchback tread is engineered to be negotiable.
- Adaptive-fetch vs. the "no caching / caller loads data" policy — reconcile.
- Confirm the demanding/alpine flags against recorded GPX (do people’s tracks show the
  same 11 m pitches, or route around them?).

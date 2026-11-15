// When a target can be photographed from the observing site, and how much integration it wants.
// Low-precision formulas (Sun to ~0.01 deg, Moon phase to a few %), plenty for "which months, how many hours".

import type { AtlasObject } from "./format"

export type Site = { name: string; lat: number; lon: number }
export const SITE_DEFAULT: Site = { name: "Rieti, Italia", lat: 42.404, lon: 12.857 }

const RAD = Math.PI / 180
const jdOf = (t: number) => t / 86400000 + 2440587.5

function sunRaDec(jd: number): [number, number] {
  const n = jd - 2451545
  const L = 280.46 + 0.9856474 * n
  const g = (357.528 + 0.9856003 * n) * RAD
  const lam = (L + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * RAD
  const eps = (23.439 - 4e-7 * n) * RAD
  return [Math.atan2(Math.cos(eps) * Math.sin(lam), Math.cos(lam)) / RAD, Math.asin(Math.sin(eps) * Math.sin(lam)) / RAD]
}

const lstOf = (jd: number, lon: number) => 280.46061837 + 360.98564736629 * (jd - 2451545) + lon

function altitude(ra: number, dec: number, lat: number, lst: number) {
  const h = (lst - ra) * RAD
  const s = Math.sin(lat * RAD) * Math.sin(dec * RAD) + Math.cos(lat * RAD) * Math.cos(dec * RAD) * Math.cos(h)
  return Math.asin(Math.max(-1, Math.min(1, s))) / RAD
}

/** illuminated fraction of the Moon (0..1) at time t */
export function moonLit(t: number) {
  const P = 29.530588853
  const age = (((jdOf(t) - 2451550.1) % P) + P) % P
  return (1 - Math.cos((2 * Math.PI * age) / P)) / 2
}

export const MIN_ALT = 30 // degrees: lower than this the sky is thick and the gradients strong

export type Night = {
  /** hours of astronomical night (Sun below -18 deg) */
  dark: number
  /** hours of astronomical night with the target above MIN_ALT */
  good: number
  /** first and last good moment (ms), when good > 0 */
  from?: number
  to?: number
  /** time of the highest point during the night, and that altitude */
  best?: number
  bestAlt: number
}

/** the night that starts at local noon of `day` (a Date in the browser's time zone) */
export function night(o: { ra: number; dec: number }, site: Site, day: Date): Night {
  const t0 = new Date(day.getFullYear(), day.getMonth(), day.getDate(), 12).getTime()
  const step = 10 * 60000
  let dark = 0
  let good = 0
  let from: number | undefined
  let to: number | undefined
  let best: number | undefined
  let bestAlt = -90
  for (let t = t0; t < t0 + 86400000; t += step) {
    const jd = jdOf(t)
    const lst = lstOf(jd, site.lon)
    const [sra, sdec] = sunRaDec(jd)
    if (altitude(sra, sdec, site.lat, lst) > -18) continue
    dark += step
    const a = altitude(o.ra, o.dec, site.lat, lst)
    if (a > bestAlt) [bestAlt, best] = [a, t]
    if (a >= MIN_ALT) {
      good += step
      from ??= t
      to = t
    }
  }
  return { dark: dark / 3600000, good: good / 3600000, from, to, best, bestAlt }
}

export const MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]

export type Visibility = {
  /** highest altitude it ever reaches from the site */
  culmination: number
  /** good hours per night, on the 15th of each month of this year */
  months: number[]
  /** indices of the best months (at least 75% of the best one), [] when it never gets above MIN_ALT at night */
  bestMonths: number[]
  tonight: Night
  moon: number
}

export function visibility(o: { ra: number; dec: number }, site: Site, now = new Date()): Visibility {
  const y = now.getFullYear()
  const months = MONTHS.map((_, m) => night(o, site, new Date(y, m, 15)).good)
  const top = Math.max(...months)
  const ok = months.map((h) => top > 0.4 && h >= top * 0.75)
  // a single month in between two good ones counts too (June: short nights, but the season goes on)
  const bestMonths = ok.map((v, m) => (v || (ok[(m + 11) % 12] && ok[(m + 1) % 12]) ? m : -1)).filter((m) => m >= 0)
  let culmination = 90 - Math.abs(site.lat - o.dec)
  if (culmination > 90) culmination = 180 - culmination
  return { culmination, months, bestMonths, tonight: night(o, site, now), moon: moonLit(now.getTime()) }
}

/** "ottobre – dicembre", "dicembre – febbraio", "marzo, giugno – luglio" */
export function monthRange(ms: number[]): string {
  if (!ms.length) return ""
  const set = new Set(ms)
  // runs of consecutive months, wrapping around the year
  const start = ms.find((m) => !set.has((m + 11) % 12))
  if (start === undefined) return "tutto l'anno"
  const runs: [number, number][] = []
  for (let k = 0, m = start; k < 12; k++, m = (m + 1) % 12) {
    if (!set.has(m)) continue
    if (runs.length && (runs[runs.length - 1][1] + 1) % 12 === m) runs[runs.length - 1][1] = m
    else runs.push([m, m])
  }
  return runs.map(([a, b]) => (a === b ? MONTHS[a] : `${MONTHS[a]} – ${MONTHS[b]}`)).join(", ")
}

export const fmtClock = (t?: number) => (t === undefined ? "—" : new Date(t).toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" }))
export const fmtHours = (h: number) => `${h.toLocaleString("it-IT", { maximumFractionDigits: 1 })} h`

// ------------------------------------------------------------------ how much integration

export type Advice = {
  /** hours for a good result, and for a very good one */
  good: number
  great: number
  tips: string[]
}

// hours for a good result with a small refractor and a cooled camera, from a moderately dark sky
const BASE: Record<string, number> = { galaxy: 8, globular: 2, cluster: 1, nebula: 5, dark: 10, remnant: 8, planetary: 3, star: 1, other: 4 }
// typical surface brightness (mag/arcsec^2) behind those hours, for the kinds whose catalogue magnitude is reliable
const SB_REF: Record<string, number> = { galaxy: 22, globular: 19.5, planetary: 19 }

const nice = (h: number) => (h < 2 ? Math.round(h * 2) / 2 : h < 10 ? Math.round(h) : Math.round(h / 5) * 5)

export function advice(o: AtlasObject): Advice {
  const kind = o.kind in BASE ? o.kind : "other"
  let h = BASE[kind]
  const tips: string[] = []
  const maj = o.size_arcmin ?? 0
  const min = o.size2_arcmin ?? maj
  let sb: number | null = null
  if (o.mag != null && maj > 0 && SB_REF[kind] !== undefined) {
    sb = o.mag + 2.5 * Math.log10((Math.PI / 4) * maj * min * 3600)
    h *= Math.min(2.5, Math.max(0.5, Math.pow(10, 0.4 * 0.6 * (sb - SB_REF[kind]))))
  }
  const reflection = o.type === "RfN"
  if (reflection) h = 6
  const good = Math.max(1, nice(h))
  const great = nice(Math.max(good * 2.5, good + 2))

  if (kind === "nebula" && !reflection)
    tips.push("Nebulosa a emissione: con un filtro duale Hα/OIII (o in banda stretta con la mono) si fa anche da cielo di città e con la Luna.")
  if (reflection) tips.push("Nebulosa a riflessione: banda larga, niente filtri a banda stretta; serve cielo buio e senza Luna.")
  if (kind === "remnant") tips.push("Resto di supernova: Hα e OIII deboli, conviene la banda stretta e tanto tempo sull'OIII.")
  if (kind === "galaxy") tips.push("Galassia: banda larga (colore senza filtri o LRGB), cielo buio e senza Luna; il filtro duale non aiuta.")
  if (kind === "galaxy" && o.mag != null && o.mag < 7) tips.push("Nucleo molto luminoso: aggiungi pose corte per non bruciarlo (HDR).")
  if (kind === "globular") tips.push("Ammasso globulare: pose brevi (60–120 s) per non saturare il centro; conta più il seeing del tempo totale.")
  if (kind === "cluster") tips.push("Ammasso aperto: bastano poche ore; pose brevi per tenere i colori delle stelle.")
  if (kind === "planetary") tips.push("Nebulosa planetaria: piccola e luminosa; Hα e OIII aiutano, pose brevi se il seeing è buono.")
  if (kind === "dark") tips.push("Nebulosa oscura: cielo molto buio e niente Luna; il segnale è debole, servono tante ore.")
  if (sb !== null && sb > 23) tips.push(`Luminosità superficiale bassa (${sb.toFixed(1)} mag/″²): più tempo del solito.`)
  if (maj > 0 && maj < 8) tips.push(`Piccolo (${maj.toFixed(maj < 3 ? 1 : 0)}′): serve focale lunga (800 mm o più) o un forte ritaglio.`)
  if (maj > 100) tips.push(`Grande (${(maj / 60).toFixed(1)}°): focale corta (400 mm o meno) o un mosaico.`)
  return { good, great, tips }
}

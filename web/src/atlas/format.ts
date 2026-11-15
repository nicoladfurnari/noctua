export type Session = {
  file: string
  kind: "stack" | "final" | string
  when: string
  date: string
  exposure_s: number | null
  frames: number | null
  filters: string[]
  thumb: string
}

export type AtlasObject = {
  id: string
  key?: string
  name: string
  names: string[]
  common: string
  type: string
  kind: string
  ra: number
  dec: number
  l: number
  b: number
  ly: number
  known: boolean
  xyz: [number, number, number]
  size_arcmin: number | null
  captured: boolean
  sessions?: Session[]
  thumb?: string
  exposure_s?: number
  frames?: number
  first?: string
}

export type AtlasScene = {
  objects: AtlasObject[]
  stats: { targets: number; results: number; exposure_s: number; nearest: string; farthest: string; catalogue: number }
}

const nf = (v: number, d = 0) => v.toLocaleString("it-IT", { maximumFractionDigits: d })

/** 1 344 al · 22 mila al · 2,54 milioni di al */
export function fmtLy(ly: number, long = false): string {
  const u = long ? "anni luce" : "al"
  if (ly < 1e4) return `${nf(Math.round(ly))} ${u}`
  if (ly < 1e6) return `${nf(ly / 1e3, ly < 1e5 ? 1 : 0)} mila ${u}`
  if (ly < 1e9) return `${nf(ly / 1e6, ly < 1e7 ? 2 : ly < 1e8 ? 1 : 0)} milioni di ${u}`
  return `${nf(ly / 1e9, 2)} miliardi di ${u}`
}

/** compact form for small readouts: 1 344 al · 22 mila al · 28 M al */
export function fmtLyShort(ly: number): string {
  if (ly < 1e4) return `${nf(Math.round(ly))} al`
  if (ly < 1e6) return `${nf(ly / 1e3, 0)} mila al`
  if (ly < 1e9) return `${nf(ly / 1e6, ly < 1e7 ? 1 : 0)} M al`
  return `${nf(ly / 1e9, 1)} G al`
}

/** when the light collected by the sensor left the object */
export function lightLeft(ly: number, now = new Date().getFullYear()): string {
  const year = now - ly
  if (year > 0) return `partita intorno al ${Math.round(year / 10) * 10} d.C.`
  if (ly < 12000) return `partita intorno al ${nf(Math.round(-year / 100) * 100)} a.C.`
  if (ly < 3e5) return `partita ${nf(ly / 1e3, 0)} mila anni fa, con l'uomo ancora nelle caverne`
  if (ly < 2.6e6) return `partita ${fmtAgo(ly)}, prima che esistesse Homo sapiens`
  if (ly < 6.6e7) return `partita ${fmtAgo(ly)}, dopo la scomparsa dei dinosauri`
  if (ly < 2.3e8) return `partita ${fmtAgo(ly)}, quando sulla Terra c'erano i dinosauri`
  return `partita ${fmtAgo(ly)}, prima dei dinosauri`
}
const fmtAgo = (ly: number) => (ly < 1e6 ? `${nf(ly / 1e3, 0)} mila anni fa` : `${nf(ly / 1e6, ly < 1e7 ? 1 : 0)} milioni di anni fa`)

export function fmtExposure(s: number | null | undefined): string {
  if (!s) return "—"
  const h = Math.floor(s / 3600)
  const m = Math.round((s - h * 3600) / 60)
  return h ? `${h} h ${String(m).padStart(2, "0")} min` : `${m} min`
}

export function fmtRaDec(ra: number, dec: number): string {
  const h = ra / 15
  const hh = Math.floor(h)
  const mm = Math.floor((h - hh) * 60)
  const sign = dec < 0 ? "−" : "+"
  const d = Math.abs(dec)
  const dd = Math.floor(d)
  const dm = Math.round((d - dd) * 60)
  return `${String(hh).padStart(2, "0")}h ${String(mm).padStart(2, "0")}m · ${sign}${String(dd).padStart(2, "0")}° ${String(dm).padStart(2, "0")}′`
}

export const KIND_IT: Record<string, string> = {
  galaxy: "Galassia",
  globular: "Ammasso globulare",
  cluster: "Ammasso aperto",
  nebula: "Nebulosa",
  dark: "Nebulosa oscura",
  remnant: "Resto di supernova",
  planetary: "Nebulosa planetaria",
  star: "Stella",
  other: "Oggetto",
}

export const KIND_COLOR: Record<string, string> = {
  galaxy: "#8fb4ff",
  globular: "#f0c070",
  cluster: "#d9d4c8",
  nebula: "#ff8f7a",
  dark: "#8a8478",
  remnant: "#c9a0ff",
  planetary: "#7fe0c0",
  star: "#ffffff",
  other: "#a8a296",
}

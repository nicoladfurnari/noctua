import * as React from "react"
import { Check, ChevronRight, CircleHelp, Crosshair, Home, ImagePlus, Layers as LayersIcon, Maximize, Minus, Plus, RefreshCw, Route, Search, Trash2, X } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn } from "@/lib/utils"

import { fmtExposure, fmtLy, fmtLyShort, fmtRaDec, KIND_COLOR, KIND_IT, lightLeft, type AtlasObject, type AtlasScene } from "./format"
import { Plan, SkyPreview } from "./Plan"
import { GALACTIC_CENTRE, photoSize, Scene, type Goal, type Layers, type View } from "./Scene"
import { SITE_DEFAULT, type Site } from "./sky"

const LAYERS: [key: keyof Layers, label: string][] = [
  ["labels", "Nomi dei tuoi bersagli"],
  ["messier", "Nomi dei Messier da fare"],
  ["photos", "Le tue foto"],
  ["routes", "Rotte dal Sole"],
  ["rings", "Anelli di distanza"],
  ["catalogue", "Oggetti del catalogo"],
  ["galaxy", "Via Lattea e galassie"],
]

// the regions crossed while zooming out, by the distance of the camera from the point it looks at (light years)
const REALMS: [upTo: number, name: string][] = [
  [3e3, "Dintorni del Sole"],
  [2.5e4, "Braccio di Orione"],
  [4e5, "Via Lattea"],
  [1.5e7, "Gruppo Locale"],
  [4e8, "Superammasso della Vergine"],
  [Infinity, "Universo vicino"],
]
export const realmOf = (dist: number) => REALMS.find(([up]) => dist < up)![1]
const LOG_MIN = 1.3 // 20 light years
const LOG_MAX = 9.3 // 2 billion

/** Vertical scale, logarithmic: drag (or click) to change the distance; the ticks are where a region ends. */
function ScaleSlider({ dist, onChange }: { dist: number; onChange: (d: number) => void }) {
  const ref = React.useRef<HTMLDivElement>(null)
  const pos = (d: number) => 1 - (Math.min(LOG_MAX, Math.max(LOG_MIN, Math.log10(d))) - LOG_MIN) / (LOG_MAX - LOG_MIN)
  const at = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect()
    const k = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))
    onChange(Math.pow(10, LOG_MIN + (1 - k) * (LOG_MAX - LOG_MIN)))
  }
  return (
    <div
      ref={ref}
      className="atlas-slider"
      title="Scala: trascina per allontanarti o avvicinarti"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        at(e)
      }}
      onPointerMove={(e) => {
        if (e.buttons) at(e)
      }}
    >
      {REALMS.slice(0, -1).map(([up, name]) => (
        <i key={name} style={{ top: `${pos(up) * 100}%` }} title={name} />
      ))}
      <b style={{ top: `${pos(dist) * 100}%` }} />
    </div>
  )
}

function Tool({ on, className, ...p }: React.ComponentProps<"button"> & { on?: boolean }) {
  return (
    <button
      type="button"
      {...p}
      className={cn(
        "atlas-tool tw:flex tw:size-[38px] tw:items-center tw:justify-center tw:text-foreground",
        on && "on",
        className
      )}
    />
  )
}

export type AtlasProps = {
  scene: AtlasScene | null
  active: boolean
  thumbUrl: (name: string) => string
  /** an object to open on (its name, e.g. "M42"): selected and approached once the scene is there */
  focus?: string
  onAdd?: () => void
  onRefresh?: () => void
  onRemove?: (key: string, file?: string) => void
  onOpen?: (file: string) => void
}

const LEVELS: { label: string; realm: string; hint: string; goal: Goal }[] = [
  { label: "Universo vicino", realm: "Universo vicino", hint: "fino agli ammassi lontani", goal: { target: [0, 0, 0], dist: 1.4e9 } },
  { label: "Vergine", realm: "Superammasso della Vergine", hint: "le galassie di Messier", goal: { target: [0, 0, 0], dist: 1.7e8 } },
  { label: "Gruppo Locale", realm: "Gruppo Locale", hint: "Andromeda, Triangolo", goal: { target: [0, 0, 0], dist: 8.5e6 } },
  { label: "Via Lattea", realm: "Via Lattea", hint: "tutta la galassia", goal: { target: GALACTIC_CENTRE, dist: 190000 } },
  { label: "Braccio di Orione", realm: "Braccio di Orione", hint: "le nebulose intorno al Sole", goal: { target: [0, 0, 0], dist: 9000 } },
  { label: "Sole", realm: "Dintorni del Sole", hint: "i dintorni del Sole", goal: { target: [0, 0, 0], dist: 1500 } },
]
const HELP: [what: string, how: string][] = [
  ["Vai su un oggetto", "clic sull'oggetto, sull'elenco o cerca in alto"],
  ["Ruota", "trascina"],
  ["Zoom", "rotella (verso il cursore; si ferma sull'oggetto) · + e −"],
  ["Sposta", "tasto destro e trascina"],
  ["Torna al Sole", "tasto casa · H"],
  ["Schermo intero", "F"],
  ["Lascia l'oggetto", "Esc"],
]
const LAYERS_KEY = "noctua_atlas_layers"
const LAYERS_DEFAULT: Layers = { labels: true, messier: true, routes: true, rings: true, catalogue: true, galaxy: true, photos: true }

function niceScale(lyPerPx: number) {
  // a bar between 70 and 160 px long standing for 1, 2 or 5 x 10^n light years
  const raw = lyPerPx * 110
  const p = Math.pow(10, Math.floor(Math.log10(raw)))
  const m = raw / p
  const v = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p
  return { ly: v, px: v / lyPerPx }
}

function Btn({ className, ...p }: React.ComponentProps<"button">) {
  return (
    <button
      type="button"
      {...p}
      className={cn(
        "atlas-btn tw:inline-flex tw:items-center tw:gap-1.5 tw:rounded-md tw:border tw:border-input tw:bg-background/70 tw:px-2.5 tw:py-1.5 tw:text-[12.5px] tw:text-foreground tw:backdrop-blur tw:transition-colors tw:hover:bg-accent tw:disabled:opacity-40",
        className
      )}
    />
  )
}

function Row({ o, on, onClick, onDouble, thumbUrl }: { o: AtlasObject; on: boolean; onClick: () => void; onDouble?: () => void; thumbUrl: (n: string) => string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      onDoubleClick={onDouble}
      className={cn(
        "atlas-row tw:flex tw:w-full tw:items-center tw:gap-2.5 tw:border-0 tw:border-b tw:border-border tw:bg-transparent tw:px-3 tw:py-2 tw:text-left tw:text-foreground tw:transition-colors tw:hover:bg-accent/60",
        on && "tw:bg-accent"
      )}
    >
      {o.captured ? (
        o.thumb ? (
          <img src={thumbUrl(o.thumb)} alt="" className="tw:size-10 tw:shrink-0 tw:rounded-sm tw:border tw:object-cover" />
        ) : (
          <span className="tw:size-10 tw:shrink-0 tw:rounded-sm tw:border tw:bg-black" />
        )
      ) : (
        <i className="tw:ml-1 tw:inline-block tw:size-2 tw:shrink-0 tw:rounded-full" style={{ background: KIND_COLOR[o.kind] ?? KIND_COLOR.other }} />
      )}
      <span className="tw:min-w-0 tw:flex-1">
        <span className="tw:block tw:truncate tw:text-[13px] tw:font-medium">
          {o.name}
          {o.common && <span className="tw:ml-1.5 tw:font-normal tw:text-muted-foreground">{o.common}</span>}
        </span>
        <span className="tw:block tw:truncate tw:font-mono tw:text-[11.5px] tw:text-muted-foreground tw:tabular-nums">
          {fmtLy(o.ly)}
          {!o.known && " · stimata"}
          {o.captured && o.exposure_s ? ` · ${fmtExposure(o.exposure_s)}` : ""}
        </span>
      </span>
    </button>
  )
}

export function Atlas({ scene, active, thumbUrl, focus, onAdd, onRefresh, onRemove, onOpen }: AtlasProps) {
  const objects = scene?.objects ?? []
  const got = React.useMemo(() => objects.filter((o) => o.captured).sort((a, b) => a.ly - b.ly), [objects])
  const byId = React.useMemo(() => new Map(objects.map((o) => [o.id, o])), [objects])
  const [selected, setSelected] = React.useState<string | null>(null)
  const [hovered, setHovered] = React.useState<string | null>(null)
  const [goal, setGoal] = React.useState<Goal | null>(null)
  const [dist, setDist] = React.useState(9000)
  const [query, setQuery] = React.useState("")
  const [tab, setTab] = React.useState("got")
  const box = React.useRef<HTMLDivElement>(null)
  const [height, setHeight] = React.useState(700)

  // the first view frames what has been photographed inside the galaxy (or the neighbourhood when empty)
  const start = React.useMemo<View>(() => {
    const near = got.filter((o) => o.ly < 60000)
    const far = near.length ? Math.max(...near.map((o) => o.ly)) : 3500
    return { target: [0, 0, 0], dist: Math.min(Math.max(far * 2.6, 6000), 220000) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene === null])

  React.useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(() => setHeight(el.clientHeight || 700))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  React.useEffect(() => {
    if (selected && !byId.has(selected)) setSelected(null)
  }, [byId, selected])

  const sel = selected ? byId.get(selected) ?? null : null
  const [site, setSite] = React.useState<Site>(scene?.site ?? SITE_DEFAULT)
  React.useEffect(() => {
    if (scene?.site) setSite(scene.site)
  }, [scene?.site?.lat, scene?.site?.lon, scene?.site?.name]) // eslint-disable-line react-hooks/exhaustive-deps
  const saveSite = async (s: Site) => {
    const r = await fetch("/api/atlas/site", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s) })
    if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `errore ${r.status}`)
    setSite(await r.json())
  }
  const frame = (o: AtlasObject) => setGoal({ target: [o.xyz[0] / 2, o.xyz[1] / 2, o.xyz[2] / 2], dist: Math.max(o.ly * 1.35, 400) })
  // close enough for the photograph (or the object at its real size) to fill a good part of the view
  const approach = (o: AtlasObject) => setGoal({ target: o.xyz, dist: Math.max(photoSize(o) * 2.4, 6) })
  const pick = React.useCallback(
    (id: string | null) => {
      setSelected(id)
      const o = id ? byId.get(id) : null
      if (o) approach(o)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [byId]
  )
  const [layers, setLayers] = React.useState<Layers>(() => {
    try {
      return { ...LAYERS_DEFAULT, ...JSON.parse(localStorage.getItem(LAYERS_KEY) || "{}") }
    } catch {
      return LAYERS_DEFAULT
    }
  })
  React.useEffect(() => {
    try {
      localStorage.setItem(LAYERS_KEY, JSON.stringify(layers))
    } catch {
      /* private window: the choice lasts for this visit only */
    }
  }, [layers])
  const [layersOpen, setLayersOpen] = React.useState(false)
  const [helpOpen, setHelpOpen] = React.useState(false)
  const [find, setFind] = React.useState("")
  const found = React.useMemo(() => {
    const k = find.trim().toLowerCase().replace(/\s+/g, "")
    if (!k) return []
    const hit = (o: AtlasObject) => o.names.some((n) => n.toLowerCase().replace(/\s+/g, "").includes(k)) || o.common.toLowerCase().replace(/\s+/g, "").includes(k)
    const exact = (o: AtlasObject) => o.names.some((n) => n.toLowerCase().replace(/\s+/g, "") === k)
    return objects.filter(hit).sort((a, b) => Number(exact(b)) - Number(exact(a)) || Number(b.captured) - Number(a.captured) || a.ly - b.ly).slice(0, 8)
  }, [find, objects])
  // the address remembers the object looked at: reloading or sharing the link comes back to it
  React.useEffect(() => {
    if (!active || !scene) return
    const o = selected ? byId.get(selected) : null
    const hash = o ? "#atlante/" + encodeURIComponent(o.name) : "#atlante"
    if (location.hash !== hash) history.replaceState(null, "", location.pathname + location.search + hash)
  }, [active, scene, selected, byId])
  const home = React.useCallback(() => {
    setSelected(null)
    setGoal({ ...start })
  }, [start])
  const fullscreen = React.useCallback(() => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void box.current?.requestFullscreen?.()
  }, [])
  // keys, while the atlas is the page shown: + / - zoom, Home or H the Sun, F full screen, Esc leaves the object
  React.useEffect(() => {
    if (!active) return
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement | null)?.closest?.("input, textarea, select") || e.ctrlKey || e.altKey || e.metaKey) return
      if (e.key === "+" || e.key === "=") setGoal({ zoom: 0.5 })
      else if (e.key === "-" || e.key === "_") setGoal({ zoom: 2 })
      else if (e.key === "Home" || e.key === "h" || e.key === "H") home()
      else if (e.key === "f" || e.key === "F") fullscreen()
      else if (e.key === "Escape" && !document.fullscreenElement) setSelected(null)
      else return
      e.preventDefault()
    }
    window.addEventListener("keydown", key)
    return () => window.removeEventListener("keydown", key)
  }, [active, home, fullscreen])
  const onArrive = React.useCallback(() => setGoal(null), [])
  const flyTo = React.useCallback(
    (id: string) => {
      const o = byId.get(id)
      if (!o) return
      setSelected(id)
      approach(o)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [byId]
  )
  const focused = React.useRef(false)
  React.useEffect(() => {
    if (focused.current || !focus || !objects.length) return
    const key = focus.toLowerCase().replace(/\s+/g, "")
    const o = objects.find((x) => [x.id, ...x.names].some((n) => n.toLowerCase().replace(/\s+/g, "") === key))
    focused.current = true
    if (o) {
      setSelected(o.id)
      approach(o)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus, objects])
  const scale = niceScale((2 * dist * Math.tan((42 * Math.PI) / 360)) / Math.max(height, 200))
  const q = query.trim().toLowerCase()
  const catalogue = React.useMemo(
    () => objects.filter((o) => !q || o.names.some((n) => n.toLowerCase().replace(/\s+/g, "").includes(q.replace(/\s+/g, ""))) || o.common.toLowerCase().includes(q)),
    [objects, q]
  )
  const farthest = got.length ? got[got.length - 1] : null

  return (
    <div ref={box} className={cn("nui atlas tw:relative tw:size-full tw:overflow-hidden tw:rounded-md tw:border tw:select-none", dist > 45000 && "far")}>
      <div className="atlas-vignette" />
      <div className="tw:absolute tw:inset-y-0 tw:right-0 tw:left-[284px]">
        {scene && (
          <Scene objects={objects} active={active} goal={goal} hovered={hovered} selected={selected} onHover={setHovered} onSelect={pick} onApproach={flyTo} onArrive={onArrive} onDistance={setDist} start={start} layers={layers} thumbUrl={thumbUrl} />
        )}
      </div>
      <div className="atlas-frame" />

      {/* left: what has been photographed, and the catalogue */}
      <div className="tw:absolute tw:top-3 tw:bottom-3 tw:left-3 tw:z-50 tw:flex tw:w-[272px] tw:flex-col tw:rounded-md tw:border tw:bg-background/80 tw:backdrop-blur-md">
        <div className="tw:border-b tw:px-3 tw:pt-3 tw:pb-2.5">
          <div className="tw:flex tw:items-center tw:justify-between">
            <span className="tw:text-[11.5px] tw:font-semibold tw:tracking-[0.18em] tw:text-muted-foreground">ATLANTE</span>
            <Badge variant="outline" className="tw:font-mono tw:text-[10.5px] tw:text-muted-foreground">
              SOLE · 0 al
            </Badge>
          </div>
          <div className="tw:mt-2.5 tw:grid tw:grid-cols-3 tw:gap-2">
            {[
              [String(scene?.stats.targets ?? 0), "bersagli"],
              [fmtExposure(scene?.stats.exposure_s).replace(" min", "′").replace(" h ", "h "), "integrati"],
              [farthest ? fmtLyShort(farthest.ly) : "—", "più lontano"],
            ].map(([v, l]) => (
              <div key={l}>
                <div className="tw:truncate tw:font-mono tw:text-[15px] tw:leading-tight tw:tabular-nums">{v}</div>
                <div className="tw:text-[10.5px] tw:tracking-wide tw:text-muted-foreground tw:uppercase">{l}</div>
              </div>
            ))}
          </div>
        </div>
        <Tabs value={tab} onValueChange={setTab} className="tw:min-h-0 tw:flex-1 tw:gap-0">
          <TabsList className="tw:m-2 tw:grid tw:h-8 tw:grid-cols-2 tw:bg-muted">
            <TabsTrigger value="got" className="atlas-tab tw:text-[12.5px]">
              Fotografati {got.length ? `· ${got.length}` : ""}
            </TabsTrigger>
            <TabsTrigger value="cat" className="atlas-tab tw:text-[12.5px]">
              Catalogo
            </TabsTrigger>
          </TabsList>
          <TabsContent value="got" className="tw:min-h-0 tw:flex-1 tw:overflow-y-auto">
            {got.length ? (
              got.map((o) => <Row key={o.id} o={o} on={o.id === selected} onClick={() => pick(o.id)} onDouble={() => flyTo(o.id)} thumbUrl={thumbUrl} />)
            ) : (
              <p className="tw:m-0 tw:px-3 tw:py-4 tw:text-[12.5px] tw:leading-relaxed tw:text-muted-foreground">
                Ancora nessun bersaglio. Al termine di uno stacking o di un export il soggetto compare qui e sulla mappa, alla sua distanza
                reale. Per le immagini già fatte usa <b className="tw:text-foreground">Aggiungi immagine</b>.
              </p>
            )}
          </TabsContent>
          <TabsContent value="cat" className="tw:flex tw:min-h-0 tw:flex-1 tw:flex-col">
            <label className="tw:mx-2 tw:mb-1.5 tw:flex tw:items-center tw:gap-2 tw:rounded-md tw:border tw:border-input tw:px-2 tw:text-muted-foreground">
              <Search className="tw:size-3.5" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="M 31, NGC 7000, Velo…"
                className="atlas-input tw:h-7 tw:min-w-0 tw:flex-1 tw:border-0 tw:bg-transparent tw:p-0 tw:text-[12.5px] tw:text-foreground tw:outline-none"
              />
            </label>
            <div className="tw:min-h-0 tw:flex-1 tw:overflow-y-auto">
              {catalogue.map((o) => (
                <Row key={o.id} o={o} on={o.id === selected} onClick={() => pick(o.id)} onDouble={() => flyTo(o.id)} thumbUrl={thumbUrl} />
              ))}
            </div>
          </TabsContent>
        </Tabs>
        <div className="tw:flex tw:gap-2 tw:border-t tw:p-2">
          <Btn className="tw:flex-1 tw:justify-center" onClick={onAdd}>
            <ImagePlus className="tw:size-3.5" />
            Aggiungi immagine
          </Btn>
          <Btn onClick={onRefresh} title="Rileggi l'atlante">
            <RefreshCw className="tw:size-3.5" />
          </Btn>
        </div>
      </div>

      {/* top: where to look */}
      <div className="tw:absolute tw:top-3 tw:left-[296px] tw:z-[60] tw:w-[min(430px,calc(100%-700px))] tw:min-w-[240px]">
        <label className="tw:flex tw:items-center tw:gap-2 tw:rounded-md tw:border tw:border-input tw:bg-background/80 tw:px-2.5 tw:text-muted-foreground tw:backdrop-blur-md">
          <Search className="tw:size-4 tw:shrink-0" />
          <input
            value={find}
            onChange={(e) => setFind(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && found[0]) {
                pick(found[0].id)
                setFind("")
              } else if (e.key === "Escape") setFind("")
            }}
            placeholder="Cerca un oggetto: M 31, NGC 7000, Velo…"
            className="atlas-input tw:h-8 tw:min-w-0 tw:flex-1 tw:border-0 tw:bg-transparent tw:p-0 tw:text-[13px] tw:text-foreground tw:outline-none"
          />
          <span className="tw:shrink-0 tw:font-mono tw:text-[11px] tw:tabular-nums" title="oggetti sulla mappa">
            {objects.length}
          </span>
        </label>
        {found.length > 0 && (
          <div className="tw:mt-1 tw:overflow-hidden tw:rounded-md tw:border tw:bg-background/95 tw:backdrop-blur-md">
            {found.map((o) => (
              <Row
                key={o.id}
                o={o}
                on={false}
                onClick={() => {
                  pick(o.id)
                  setFind("")
                }}
                thumbUrl={thumbUrl}
              />
            ))}
          </div>
        )}
      </div>

      {/* where you are: the scales from the largest to the Sun, then the object the view turns around */}
      <div className="tw:absolute tw:top-[52px] tw:right-[60px] tw:left-[298px] tw:z-50 tw:flex tw:flex-wrap tw:items-center tw:gap-x-1 tw:gap-y-0.5 tw:font-mono tw:text-[11.5px] tw:tracking-wide tw:text-muted-foreground tw:uppercase">
        {LEVELS.map((l, i) => (
          <React.Fragment key={l.label}>
            {i > 0 && <ChevronRight className="tw:size-3 tw:opacity-60" />}
            <button
              type="button"
              title={l.hint}
              onClick={() => {
                if (l.label === "Sole") setSelected(null)
                setGoal(l.goal)
              }}
              className={cn(
                "atlas-x tw:rounded tw:border-0 tw:bg-transparent tw:px-1 tw:py-0.5 tw:uppercase tw:tracking-wide tw:text-inherit tw:hover:text-foreground",
                !sel && realmOf(dist) === l.realm && "tw:text-foreground tw:underline tw:decoration-destructive tw:decoration-2 tw:underline-offset-4"
              )}
            >
              {l.label}
            </button>
          </React.Fragment>
        ))}
        {sel && (
          <>
            <ChevronRight className="tw:size-3 tw:opacity-60" />
            <span className="tw:px-1 tw:text-foreground tw:underline tw:decoration-destructive tw:decoration-2 tw:underline-offset-4">{sel.name}</span>
            <span className="tw:normal-case">· {fmtLy(sel.ly)}</span>
          </>
        )}
      </div>

      {/* right: zoom, home, layers, full screen */}
      <div className="tw:absolute tw:right-3 tw:bottom-3 tw:z-50 tw:flex tw:items-end tw:gap-2">
        {helpOpen && (
          <div className="tw:rounded-md tw:border tw:bg-background/90 tw:px-3 tw:py-2.5 tw:backdrop-blur-md">
            <div className="tw:mb-1.5 tw:text-[10.5px] tw:tracking-[0.14em] tw:text-muted-foreground tw:uppercase">Comandi</div>
            <dl className="atlas-dl tw:mt-0!">
              {HELP.map(([what, how]) => (
                <React.Fragment key={what}>
                  <dt>{what}</dt>
                  <dd>{how}</dd>
                </React.Fragment>
              ))}
            </dl>
          </div>
        )}
        {layersOpen && (
          <div className="tw:rounded-md tw:border tw:bg-background/90 tw:p-1.5 tw:backdrop-blur-md">
            {LAYERS.map(([k, label]) => (
              <button
                key={k}
                type="button"
                onClick={() => setLayers((l) => ({ ...l, [k]: !l[k] }))}
                className="atlas-x tw:flex tw:w-full tw:items-center tw:gap-2 tw:rounded tw:border-0 tw:bg-transparent tw:px-2 tw:py-1.5 tw:text-left tw:text-[12.5px] tw:text-foreground tw:hover:bg-accent"
              >
                <span className={cn("tw:flex tw:size-4 tw:items-center tw:justify-center tw:rounded-sm tw:border tw:border-input", layers[k] && "tw:border-destructive tw:bg-destructive")}>
                  {layers[k] && <Check className="tw:size-3 tw:text-white" />}
                </span>
                {label}
              </button>
            ))}
          </div>
        )}
        <div className="tw:flex tw:flex-col tw:items-center tw:gap-2">
          <ScaleSlider dist={dist} onChange={(d) => setGoal({ dist: d })} />
        <div className="tw:flex tw:flex-col tw:gap-1.5">
          <Tool title="Avvicina (+)" onClick={() => setGoal({ zoom: 0.5 })}>
            <Plus className="tw:size-4" />
          </Tool>
          <Tool title="Allontana (−)" onClick={() => setGoal({ zoom: 2 })}>
            <Minus className="tw:size-4" />
          </Tool>
          <Tool title="Torna al Sole (Home)" onClick={home}>
            <Home className="tw:size-4" />
          </Tool>
          <Tool title="Livelli: cosa mostrare" on={layersOpen} onClick={() => (setLayersOpen((v) => !v), setHelpOpen(false))}>
            <LayersIcon className="tw:size-4" />
          </Tool>
          <Tool title="Schermo intero (F)" onClick={fullscreen}>
            <Maximize className="tw:size-4" />
          </Tool>
          <Tool title="Comandi" on={helpOpen} onClick={() => (setHelpOpen((v) => !v), setLayersOpen(false))}>
            <CircleHelp className="tw:size-4" />
          </Tool>
        </div>
        </div>
      </div>

      {/* bottom: scale and controls */}
      <div className="tw:pointer-events-none tw:absolute tw:bottom-4 tw:left-[300px] tw:z-50 tw:flex tw:items-end tw:gap-5 tw:font-mono tw:text-[11.5px] tw:text-muted-foreground tw:tabular-nums">
        <div>
          <div className="atlas-scale" style={{ width: scale.px }} />
          <div className="tw:mt-1 tw:text-foreground">{fmtLy(scale.ly)}</div>
        </div>
        <div className="tw:pb-0.5">
          <span className="tw:text-foreground">{realmOf(dist)}</span> · quota {fmtLy(dist)} · clic: vai all'oggetto · trascina: ruota · rotella: zoom
        </div>
      </div>

      {/* right: the selected object */}
      {sel && (
        <div className="tw:absolute tw:top-[78px] tw:right-3 tw:z-50 tw:max-h-[calc(100%-90px)] tw:w-[300px] tw:max-w-[calc(100%-380px)] tw:mr-[52px] tw:overflow-y-auto tw:rounded-md tw:border tw:bg-background/85 tw:backdrop-blur-md">
          {sel.captured && sel.thumb && <img src={thumbUrl(sel.thumb)} alt="" className="tw:block tw:max-h-[210px] tw:w-full tw:rounded-t-md tw:object-cover" />}
          {!sel.captured && <SkyPreview id={sel.id} />}
          <div className="tw:p-3">
            <div className="tw:flex tw:items-start tw:justify-between tw:gap-2">
              <div className="tw:min-w-0">
                <div className="tw:text-[17px] tw:leading-tight tw:font-semibold">{sel.name}</div>
                <div className="tw:truncate tw:text-[12.5px] tw:text-muted-foreground">
                  {[sel.common, ...sel.names.slice(1)].filter(Boolean).join(" · ") || KIND_IT[sel.kind]}
                </div>
              </div>
              <button type="button" className="atlas-x tw:rounded tw:border-0 tw:bg-transparent tw:p-1 tw:text-muted-foreground tw:hover:text-foreground" onClick={() => setSelected(null)} title="Chiudi">
                <X className="tw:size-4" />
              </button>
            </div>
            <div className="tw:mt-2 tw:flex tw:flex-wrap tw:gap-1.5">
              <Badge variant="outline" style={{ borderColor: KIND_COLOR[sel.kind], color: KIND_COLOR[sel.kind] }}>
                {KIND_IT[sel.kind] ?? "Oggetto"}
              </Badge>
              {sel.captured ? <Badge variant="destructive">fotografato</Badge> : <Badge variant="secondary">da fare</Badge>}
              <Badge variant="outline" className="tw:text-muted-foreground" title={sel.known ? "Valore indicativo dalla letteratura" : "Non in tabella: distanza tipica per questo tipo di oggetto"}>
                {sel.known ? "distanza da letteratura" : "distanza stimata"}
              </Badge>
            </div>
            <div className="tw:mt-3 tw:font-mono tw:text-[22px] tw:leading-none tw:tabular-nums">{fmtLy(sel.ly, true)}</div>
            <div className="tw:mt-1.5 tw:text-[12.5px] tw:leading-snug tw:text-muted-foreground">
              {sel.known ? `La luce che raccogli è ${lightLeft(sel.ly)}.` : "Distanza non in tabella: valore tipico per questo tipo di oggetto, solo indicativo."}
            </div>
            <dl className="atlas-dl">
              <dt>AR · Dec</dt>
              <dd>{fmtRaDec(sel.ra, sel.dec)}</dd>
              <dt>Galattiche</dt>
              <dd>
                l {sel.l.toFixed(1)}° · b {sel.b >= 0 ? "+" : "−"}
                {Math.abs(sel.b).toFixed(1)}°
              </dd>
              {sel.size_arcmin ? (
                <>
                  <dt>Dimensione</dt>
                  <dd>
                    {sel.size_arcmin >= 60 ? `${(sel.size_arcmin / 60).toFixed(1)}°` : `${sel.size_arcmin.toFixed(0)}′`} ·{" "}
                    {fmtLy((sel.size_arcmin / 60) * (Math.PI / 180) * sel.ly)}
                  </dd>
                </>
              ) : null}
              {sel.captured && (
                <>
                  <dt>Integrazione</dt>
                  <dd>
                    {fmtExposure(sel.exposure_s)}
                    {sel.frames ? ` · ${sel.frames} frame` : ""}
                  </dd>
                </>
              )}
            </dl>
            <div className="tw:mt-3 tw:flex tw:gap-1.5">
              <Btn className="tw:flex-1 tw:justify-center" onClick={() => frame(sel)} title="Sole e oggetto nella stessa vista: quanto è lontano da casa">
                <Route className="tw:size-3.5" />
                Rotta
              </Btn>
              <Btn className="tw:flex-1 tw:justify-center" onClick={() => approach(sel)} title="Vola vicino all'oggetto">
                <Crosshair className="tw:size-3.5" />
                Avvicinati
              </Btn>
            </div>
            <Plan o={sel} site={site} onSite={saveSite} />
            {sel.captured && sel.sessions && (
              <div className="tw:mt-3 tw:border-t tw:pt-2">
                <div className="tw:mb-1 tw:text-[10.5px] tw:tracking-[0.14em] tw:text-muted-foreground tw:uppercase">Risultati</div>
                <div className="tw:max-h-[150px] tw:overflow-y-auto">
                  {[...sel.sessions].reverse().map((s) => (
                    <div key={s.file} className="tw:flex tw:items-center tw:gap-2 tw:py-1 tw:text-[12px]">
                      <span className="tw:min-w-0 tw:flex-1">
                        <span className="tw:block tw:truncate" title={s.file}>
                          {s.kind === "final" ? "Immagine finale" : "Integrazione"} · {(s.date || s.when).slice(0, 10)}
                        </span>
                        <span className="tw:block tw:truncate tw:font-mono tw:text-[11px] tw:text-muted-foreground">
                          {s.kind === "final" ? s.file.split(/[\\/]/).pop() : `${fmtExposure(s.exposure_s)}${s.frames ? ` · ${s.frames} frame` : ""}`}
                        </span>
                      </span>
                      {s.kind !== "final" && onOpen && (
                        <Btn className="tw:px-2 tw:py-1" onClick={() => onOpen(s.file)} title="Apri in post-produzione">
                          Apri
                        </Btn>
                      )}
                      {onRemove && sel.key && (
                        <button type="button" className="atlas-x tw:rounded tw:border-0 tw:bg-transparent tw:p-1 tw:text-muted-foreground tw:hover:text-destructive" title="Togli dall'atlante (il file resta)" onClick={() => onRemove(sel.key!, s.file)}>
                          <Trash2 className="tw:size-3.5" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

import * as React from "react"
import { Bar, BarChart, Cell, XAxis, YAxis } from "recharts"
import { LocateFixed, MapPin, Moon } from "lucide-react"

import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"

import { fmtExposure, type AtlasObject } from "./format"
import { advice, fmtClock, fmtHours, MIN_ALT, monthRange, MONTHS, visibility, type Site } from "./sky"

const Head = ({ children }: { children: React.ReactNode }) => (
  <div className="tw:mb-1 tw:text-[10.5px] tw:tracking-[0.14em] tw:text-muted-foreground tw:uppercase">{children}</div>
)

/** DSS2 picture of a target not photographed yet, downloaded by the server on the first click */
export function SkyPreview({ id }: { id: string }) {
  const [state, setState] = React.useState<"loading" | "ok" | "error">("loading")
  React.useEffect(() => setState("loading"), [id])
  const src = `/api/atlas/sky?id=${encodeURIComponent(id)}`
  return (
    <div className="tw:relative tw:h-[190px] tw:overflow-hidden tw:rounded-t-md tw:bg-black">
      <img
        key={id}
        src={src}
        alt=""
        onLoad={() => setState("ok")}
        onError={() => setState("error")}
        className="tw:block tw:h-full tw:w-full tw:object-cover tw:transition-opacity tw:duration-500"
        style={{ opacity: state === "ok" ? 1 : 0 }}
      />
      {state !== "ok" && (
        <div className="tw:absolute tw:inset-0 tw:flex tw:items-center tw:justify-center tw:px-6 tw:text-center tw:text-[12px] tw:text-muted-foreground">
          {state === "loading" ? <span className="atlas-pulse">scarico l'anteprima del cielo…</span> : "anteprima non disponibile (serve Internet)"}
        </div>
      )}
      {state === "ok" && (
        <span className="tw:absolute tw:right-1.5 tw:bottom-1 tw:rounded-sm tw:bg-black/60 tw:px-1.5 tw:font-mono tw:text-[10px] tw:text-white/70">DSS2 · come apparirebbe</span>
      )}
    </div>
  )
}

const chartConfig = { h: { label: "ore", color: "#f0a640" } } satisfies ChartConfig

function SiteEditor({ site, onSave, onClose }: { site: Site; onSave: (s: Site) => Promise<void>; onClose: () => void }) {
  const [name, setName] = React.useState(site.name)
  const [lat, setLat] = React.useState(String(site.lat))
  const [lon, setLon] = React.useState(String(site.lon))
  const [msg, setMsg] = React.useState("")
  const num = (v: string) => Number(v.replace(",", "."))
  const ok = Number.isFinite(num(lat)) && Math.abs(num(lat)) <= 90 && Number.isFinite(num(lon)) && Math.abs(num(lon)) <= 180
  const field = "tw:w-full tw:rounded tw:border tw:border-input tw:bg-background/60 tw:px-2 tw:py-1 tw:text-[12px] tw:text-foreground"
  return (
    <div className="tw:mt-1.5 tw:grid tw:gap-1.5 tw:rounded-md tw:border tw:p-2">
      <input className={field} value={name} onChange={(e) => setName(e.target.value)} placeholder="Nome del luogo" />
      <div className="tw:flex tw:gap-1.5">
        <input className={field} value={lat} onChange={(e) => setLat(e.target.value)} placeholder="Latitudine (N +)" title="Latitudine in gradi, nord positivo" />
        <input className={field} value={lon} onChange={(e) => setLon(e.target.value)} placeholder="Longitudine (E +)" title="Longitudine in gradi, est positivo" />
      </div>
      <div className="tw:flex tw:items-center tw:gap-1.5">
        <button
          type="button"
          className="atlas-btn tw:inline-flex tw:items-center tw:gap-1 tw:rounded-md tw:border tw:border-input tw:bg-background/70 tw:px-2 tw:py-1 tw:text-[12px] tw:text-foreground"
          title="Usa la posizione di questo dispositivo"
          onClick={() => {
            if (!navigator.geolocation) return setMsg("posizione non disponibile")
            setMsg("cerco la posizione…")
            navigator.geolocation.getCurrentPosition(
              (p) => {
                setLat(p.coords.latitude.toFixed(4))
                setLon(p.coords.longitude.toFixed(4))
                if (name === site.name) setName("Posizione attuale")
                setMsg("")
              },
              () => setMsg("posizione negata o non disponibile"),
              { timeout: 15000 }
            )
          }}
        >
          <LocateFixed className="tw:size-3.5" />
          Qui
        </button>
        <span className="tw:flex-1 tw:text-[11px] tw:text-muted-foreground">{msg}</span>
        <button type="button" className="atlas-btn tw:rounded-md tw:border tw:border-input tw:bg-transparent tw:px-2 tw:py-1 tw:text-[12px] tw:text-muted-foreground" onClick={onClose}>
          Annulla
        </button>
        <button
          type="button"
          disabled={!ok}
          className="atlas-btn tw:rounded-md tw:border tw:border-input tw:bg-primary tw:px-2 tw:py-1 tw:text-[12px] tw:text-primary-foreground tw:disabled:opacity-40"
          onClick={() => onSave({ name: name.trim(), lat: num(lat), lon: num(lon) }).then(onClose, (e) => setMsg(String(e.message || e)))}
        >
          Salva
        </button>
      </div>
    </div>
  )
}

/** "Quanto integrare" and "Quando": hours for a good result, best months and tonight from the observing site */
export function Plan({ o, site, onSite }: { o: AtlasObject; site: Site; onSite: (s: Site) => Promise<void> }) {
  const [editing, setEditing] = React.useState(false)
  const vis = React.useMemo(() => visibility(o, site), [o.ra, o.dec, site.lat, site.lon]) // eslint-disable-line react-hooks/exhaustive-deps
  const adv = React.useMemo(() => advice(o), [o])
  const now = new Date().getMonth()
  const rows = vis.months.map((h, m) => ({ m: MONTHS[m][0].toUpperCase(), name: MONTHS[m], h: Math.round(h * 10) / 10, i: m }))
  const top = Math.max(...vis.months)
  const t = vis.tonight
  const nights = top > 0.4 ? Math.ceil(adv.good / top) : 0
  const have = o.captured ? o.exposure_s ?? 0 : 0

  return (
    <>
      <div className="tw:mt-3 tw:border-t tw:pt-2">
        <Head>Quanto integrare</Head>
        <div className="tw:flex tw:items-baseline tw:gap-3 tw:font-mono tw:tabular-nums">
          <span>
            <span className="tw:text-[20px] tw:text-foreground">{adv.good} h</span>
            <span className="tw:ml-1 tw:text-[11px] tw:text-muted-foreground">buono</span>
          </span>
          <span>
            <span className="tw:text-[20px] tw:text-foreground">{adv.great} h</span>
            <span className="tw:ml-1 tw:text-[11px] tw:text-muted-foreground">ottimo</span>
          </span>
        </div>
        {have > 0 && (
          <div className="tw:mt-1 tw:text-[12px] tw:text-muted-foreground">
            Ne hai {fmtExposure(have)}
            {have >= adv.great * 3600 ? ": già oltre l'ottimo." : have >= adv.good * 3600 ? ": buono, altre ore lo porterebbero all'ottimo." : `: ne mancano circa ${fmtHours(adv.good - have / 3600)} per un buon risultato.`}
          </div>
        )}
        {nights > 0 && !have && (
          <div className="tw:mt-1 tw:text-[12px] tw:text-muted-foreground">
            Nel mese migliore {nights === 1 ? "basta una notte" : `servono circa ${nights} notti`} ({fmtHours(top)} utili a notte).
          </div>
        )}
        <ul className="tw:mt-1.5 tw:mb-0 tw:grid tw:gap-1 tw:pl-4 tw:text-[12px] tw:leading-snug tw:text-muted-foreground">
          {adv.tips.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </div>

      <div className="tw:mt-3 tw:border-t tw:pt-2">
        <div className="tw:flex tw:items-center tw:justify-between tw:gap-2">
          <Head>Quando</Head>
          <button
            type="button"
            className="atlas-x tw:mb-1 tw:inline-flex tw:min-w-0 tw:items-center tw:gap-1 tw:rounded tw:border-0 tw:bg-transparent tw:p-0 tw:text-[11.5px] tw:text-muted-foreground tw:hover:text-foreground"
            title="Cambia il luogo di osservazione"
            onClick={() => setEditing((v) => !v)}
          >
            <MapPin className="tw:size-3.5 tw:shrink-0" />
            <span className="tw:truncate">{site.name}</span>
          </button>
        </div>
        {editing && <SiteEditor site={site} onSave={onSite} onClose={() => setEditing(false)} />}
        {vis.culmination < 0 ? (
          <div className="tw:text-[12.5px] tw:text-muted-foreground">Da qui non sorge mai.</div>
        ) : (
          <>
            <div className="tw:text-[12.5px] tw:leading-snug">
              {vis.bestMonths.length ? (
                <>
                  Meglio a <b>{monthRange(vis.bestMonths)}</b>: fino a {fmtHours(top)} a notte sopra {MIN_ALT}°.
                </>
              ) : (
                <>Resta basso: di notte non supera mai i {MIN_ALT}°.</>
              )}{" "}
              <span className="tw:text-muted-foreground">Culmina a {Math.round(vis.culmination)}°.</span>
            </div>
            <ChartContainer config={chartConfig} className="tw:mt-1.5 tw:aspect-auto tw:h-[74px] tw:w-full">
              <BarChart data={rows} margin={{ top: 4, right: 0, bottom: 0, left: 0 }} barCategoryGap={2}>
                <XAxis dataKey="m" tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 10 }} height={14} />
                <YAxis hide domain={[0, Math.max(4, top)]} />
                <ChartTooltip
                  cursor={{ fill: "var(--accent)", opacity: 0.5 }}
                  content={({ active, payload }) =>
                    active && payload?.length ? (
                      <div className="tw:rounded tw:border tw:bg-background tw:px-2 tw:py-1 tw:text-[11.5px]">
                        {payload[0].payload.name}: {fmtHours(payload[0].payload.h)} sopra {MIN_ALT}° col buio
                      </div>
                    ) : null
                  }
                />
                <Bar dataKey="h" radius={[2, 2, 0, 0]} isAnimationActive={false}>
                  {rows.map((r) => (
                    <Cell key={r.i} fill={vis.bestMonths.includes(r.i) ? "#f0a640" : "#4a5878"} stroke={r.i === now ? "#ece7dc" : "none"} strokeWidth={1} />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
            <div className="tw:mt-1.5 tw:flex tw:items-start tw:gap-1.5 tw:text-[12px] tw:leading-snug tw:text-muted-foreground">
              <Moon className="tw:mt-px tw:size-3.5 tw:shrink-0" />
              <span>
                Stanotte:{" "}
                {t.good > 0 ? (
                  <>
                    <span className="tw:text-foreground">{fmtHours(t.good)}</span> utili, {fmtClock(t.from)}–{fmtClock(t.to)}
                    {t.best !== undefined && t.bestAlt > MIN_ALT ? `, più alto alle ${fmtClock(t.best)} (${Math.round(t.bestAlt)}°)` : ""}
                  </>
                ) : t.dark > 0 ? (
                  `non sale sopra i ${MIN_ALT}° col buio`
                ) : (
                  "niente notte astronomica"
                )}
                . Luna al {Math.round(vis.moon * 100)}%
                {vis.moon > 0.6 && (o.kind === "galaxy" || o.kind === "dark" || o.type === "RfN") ? ": troppa per questo soggetto." : "."}
              </span>
            </div>
          </>
        )}
      </div>
    </>
  )
}

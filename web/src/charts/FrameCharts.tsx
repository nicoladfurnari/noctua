import * as React from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"

export type Frame = {
  file: string
  accepted: boolean
  reason?: string
  [k: string]: unknown
}

export type FrameChartsProps = {
  frames: Frame[]
  selected: number
  onSelect?: (index: number) => void
}

// one small chart per quality number of the frames; the cursor is shared (syncId), a click picks the frame
const METRICS: [key: string, label: string, decimals: number][] = [
  ["fwhm", "FWHM", 2],
  ["eccentricity", "Eccentricità", 2],
  ["stars", "Stelle", 0],
  ["snr", "SNR", 0],
  ["background", "Fondo", 4],
  ["weight", "Peso", 2],
]

const config = { value: { label: "valore", color: "var(--muted-foreground)" } } satisfies ChartConfig
const noExt = (f: string) => f.replace(/\.fits?$/i, "")

function Dot(props: any) {
  const { cx, cy, payload, selected } = props
  if (cx == null || cy == null || payload?.value == null) return null
  const on = payload.index === selected
  return (
    <g>
      <circle cx={cx} cy={cy} r={2.6} fill={payload.accepted ? "var(--chart-1)" : "var(--chart-2)"} />
      {on && <circle cx={cx} cy={cy} r={5} fill="none" stroke="var(--chart-2)" strokeWidth={1.8} />}
    </g>
  )
}

function Metric({ frames, selected, onSelect, k, label, dec }: FrameChartsProps & { k: string; label: string; dec: number }) {
  const data = React.useMemo(
    () =>
      frames.map((f, index) => {
        const v = f[k]
        const value = typeof v === "number" && Number.isFinite(v) ? v : null
        return { index, n: index + 1, value, accepted: f.accepted, file: noExt(f.file) }
      }),
    [frames, k]
  )
  const nums = data.map((d) => d.value).filter((v): v is number => v !== null)
  const [hover, setHover] = React.useState<number | null>(null)
  if (!nums.length) return null // a report without this number (older stacks)
  const lo = Math.min(...nums)
  const hi = Math.max(...nums)
  const pad = hi > lo ? (hi - lo) * 0.14 : 1
  const h = hover !== null ? data[hover] : null
  const pick = (s: any) => {
    const i = Number(s?.activeTooltipIndex ?? s?.activeIndex)
    return Number.isInteger(i) && i >= 0 && i < data.length ? i : null
  }
  return (
    <div className="tw:min-w-0">
      <div className="tw:flex tw:items-baseline tw:justify-between tw:gap-2 tw:text-[12.5px]">
        <span>{label}</span>
        <span className="tw:truncate tw:text-muted-foreground tw:tabular-nums">
          {h && h.value !== null
            ? `${h.file}: ${h.value.toFixed(dec)}${h.accepted ? "" : " (scartato)"}`
            : `${lo.toFixed(dec)} – ${hi.toFixed(dec)}`}
        </span>
      </div>
      <ChartContainer
        config={config}
        className="tw:aspect-auto tw:h-[76px] tw:w-full tw:cursor-pointer tw:rounded-md tw:border tw:bg-[#050505]"
      >
        <LineChart
          data={data}
          syncId="noctua-frames"
          margin={{ top: 9, right: 10, bottom: 7, left: 10 }}
          onMouseMove={(s: any) => setHover(pick(s))}
          onMouseLeave={() => setHover(null)}
          onClick={(s: any) => {
            const i = pick(s)
            if (i !== null) onSelect?.(i)
          }}
        >
          <CartesianGrid vertical={false} strokeDasharray="2 4" />
          <XAxis dataKey="n" hide />
          <YAxis hide domain={[lo - pad, hi + pad]} />
          <ChartTooltip cursor={{ stroke: "var(--input)" }} content={() => null} />
          <Line
            dataKey="value"
            type="linear"
            stroke="var(--color-value)"
            strokeWidth={1}
            connectNulls
            isAnimationActive={false}
            dot={<Dot selected={selected} />}
            activeDot={false}
          />
        </LineChart>
      </ChartContainer>
    </div>
  )
}

export function FrameCharts(props: FrameChartsProps) {
  if (!props.frames.length) return null
  return (
    <div className="nui tw:mb-2.5">
      <div className="tw:grid tw:grid-cols-1 tw:gap-x-3.5 tw:gap-y-2 tw:sm:grid-cols-2">
        {METRICS.map(([k, label, dec]) => (
          <Metric key={k} {...props} k={k} label={label} dec={dec} />
        ))}
      </div>
      <div className="tw:mt-2 tw:flex tw:flex-wrap tw:items-center tw:gap-x-3 tw:text-[12.5px] tw:text-muted-foreground">
        <span className="tw:inline-flex tw:items-center tw:gap-1.5">
          <i className="tw:inline-block tw:size-2 tw:rounded-full tw:bg-chart-1" />
          usato
        </span>
        <span className="tw:inline-flex tw:items-center tw:gap-1.5">
          <i className="tw:inline-block tw:size-2 tw:rounded-full tw:bg-chart-2" />
          scartato
        </span>
        <span className="tw:inline-flex tw:items-center tw:gap-1.5">
          <i className="tw:inline-block tw:size-2.5 tw:rounded-full tw:border-2 tw:border-chart-2" />
          selezionato
        </span>
        <span>· passa il mouse per leggere, clic per vedere il frame</span>
      </div>
    </div>
  )
}

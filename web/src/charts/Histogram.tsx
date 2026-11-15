import * as React from "react"
import { Area, AreaChart, XAxis, YAxis } from "recharts"

import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"

export type HistogramData = {
  /** log-scaled bins 0..1, 256 values per channel */
  r: number[]
  g: number[]
  b: number[]
  /** share of the pixels in each bin (0..1), for the readout */
  pr: number[]
  pg: number[]
  pb: number[]
}

const config = {
  r: { label: "R", color: "#ef5a48" },
  g: { label: "G", color: "#6cc96f" },
  b: { label: "B", color: "#6a9cf2" },
} satisfies ChartConfig

const pc = (v: number) => `${(100 * v).toFixed(v < 0.0005 ? 2 : 1)}%`

/** Live histogram of the preview: R, G, B overlapped, logarithmic. onHover gets the readout of the level under
 *  the cursor (null when the mouse leaves), so the page shows it next to the black / white clipping. */
export function Histogram({ data, onHover }: { data: HistogramData | null; onHover?: (text: string | null) => void }) {
  const rows = React.useMemo(
    () => (data ? data.r.map((_, i) => ({ i, r: data.r[i], g: data.g[i], b: data.b[i] })) : []),
    [data]
  )
  if (!data) return null
  const read = (s: any) => {
    const i = Number(s?.activeTooltipIndex ?? s?.activeIndex)
    if (!Number.isInteger(i) || i < 0 || i > 255) return onHover?.(null)
    onHover?.(`livello ${i}: R ${pc(data.pr[i])} · G ${pc(data.pg[i])} · B ${pc(data.pb[i])}`)
  }
  return (
    <div className="nui tw:min-w-0 tw:flex-1">
      <ChartContainer config={config} className="tw:aspect-auto tw:h-[64px] tw:w-full">
        <AreaChart data={rows} margin={{ top: 3, right: 0, bottom: 0, left: 0 }} onMouseMove={read} onMouseLeave={() => onHover?.(null)}>
          <XAxis dataKey="i" type="number" domain={[0, 255]} hide />
          <YAxis hide domain={[0, 1.04]} />
          <ChartTooltip cursor={{ stroke: "var(--input)" }} content={() => null} />
          {(["r", "g", "b"] as const).map((k) => (
            <Area
              key={k}
              dataKey={k}
              type="linear"
              stroke={`var(--color-${k})`}
              strokeWidth={1}
              fill={`var(--color-${k})`}
              fillOpacity={0.18}
              isAnimationActive={false}
              dot={false}
              activeDot={false}
            />
          ))}
        </AreaChart>
      </ChartContainer>
    </div>
  )
}

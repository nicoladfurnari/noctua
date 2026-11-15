import "./index.css"

import * as React from "react"
import { createRoot } from "react-dom/client"

import { Atlas, type AtlasProps } from "./atlas/Atlas"
import { FrameCharts, type FrameChartsProps } from "./charts/FrameCharts"
import { Histogram, type HistogramData } from "./charts/Histogram"

// React islands for the plain-JS GUI (app.js): each mount(el, props) returns { update(props), destroy() }.
// The page owns the data and the callbacks; nothing here talks to the server.
function island<P extends object>(Component: React.ComponentType<P>) {
  return (el: Element, props: P) => {
    const root = createRoot(el)
    let current = props
    const render = () => root.render(<Component {...current} />)
    render()
    return {
      update(next: Partial<P>) {
        current = { ...current, ...next }
        render()
      },
      destroy() {
        root.unmount()
      },
    }
  }
}

export const version = "0.1.0"
export const frameCharts = island<FrameChartsProps>(FrameCharts)
export const histogram = island<{ data: HistogramData | null; onHover?: (text: string | null) => void }>(Histogram)
export const atlas = island<AtlasProps>(Atlas)

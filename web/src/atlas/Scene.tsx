import * as React from "react"
import * as THREE from "three"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Html, Line, OrbitControls } from "@react-three/drei"

import { fmtLy, KIND_COLOR, type AtlasObject } from "./format"

/** The Sun is the origin; x points to the galactic centre, z to the north galactic pole. Units: light years. */
export const GALACTIC_CENTRE: [number, number, number] = [26700, 0, 0]
const RINGS = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
const PAPER = "#ece7dc"
const RED = "#e8412f"
const AMBER = "#f0a640"

/** A view: the point the camera orbits and its distance from it. */
export type View = { target: [number, number, number]; dist: number }
/** Where to fly: a new orbit centre and / or distance; zoom multiplies the current distance (+ / - buttons). */
export type Goal = Partial<View> & { zoom?: number }
/** What the scene draws (the "layers" menu). */
export type Layers = { labels: boolean; routes: boolean; rings: boolean; catalogue: boolean; galaxy: boolean; photos: boolean }

// deterministic random numbers: the galaxy looks the same at every start
function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r() + 1e-12)) * Math.cos(2 * Math.PI * r())

/** A schematic Milky Way: bulge, four logarithmic arms, thin disc. Not a map of real stars. */
function MilkyWay() {
  const geo = React.useMemo(() => {
    const r = rng(20261115)
    const N = 70000
    const pos = new Float32Array(N * 3)
    const col = new Float32Array(N * 3)
    const warm = new THREE.Color("#ffd2a0")
    const cool = new THREE.Color("#cfd9ff")
    const pink = new THREE.Color("#ff9d8c")
    const c = new THREE.Color()
    const pitch = Math.tan((13 * Math.PI) / 180)
    for (let i = 0; i < N; i++) {
      let x: number, y: number, z: number
      const u = r()
      if (u < 0.16) {
        // bulge and bar
        const rr = Math.abs(gauss(r)) * 3200
        const th = r() * Math.PI * 2
        x = rr * Math.cos(th) * 1.5
        y = rr * Math.sin(th) * 0.8
        z = gauss(r) * 1300 * Math.exp(-rr / 6000)
        c.copy(warm).multiplyScalar(0.22 + 0.25 * r())
      } else {
        const rr = Math.min(58000, 3500 - 13000 * Math.log(1 - r() * 0.985))
        const arm = Math.floor(r() * 4)
        const inArm = u < 0.8
        const spread = inArm ? 0.16 + 0.1 * r() : 1.2
        const th = (arm * Math.PI) / 2 + Math.log(rr / 3500) / pitch + gauss(r) * spread + 0.55
        x = rr * Math.cos(th)
        y = rr * Math.sin(th)
        z = gauss(r) * (280 + rr * 0.006)
        if (inArm && r() < 0.07) c.copy(pink).multiplyScalar(0.8)
        else c.copy(cool).lerp(warm, Math.max(0, 1 - rr / 30000) * 0.6).multiplyScalar(inArm ? 0.3 + 0.45 * r() : 0.14 + 0.18 * r())
      }
      pos[i * 3] = x + GALACTIC_CENTRE[0]
      pos[i * 3 + 1] = y
      pos[i * 3 + 2] = z
      col[i * 3] = c.r
      col[i * 3 + 1] = c.g
      col[i * 3 + 2] = c.b
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3))
    g.setAttribute("color", new THREE.BufferAttribute(col, 3))
    return g
  }, [])
  return (
    <points geometry={geo} frustumCulled={false}>
      <pointsMaterial size={1.25} sizeAttenuation={false} vertexColors transparent opacity={0.42} depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  )
}

/** Faint far stars on a sphere that follows the camera: depth at every zoom level. */
function Backdrop() {
  const ref = React.useRef<THREE.Points>(null)
  const geo = React.useMemo(() => {
    const r = rng(7)
    const N = 1800
    const pos = new Float32Array(N * 3)
    for (let i = 0; i < N; i++) {
      const z = r() * 2 - 1
      const th = r() * Math.PI * 2
      const s = Math.sqrt(1 - z * z)
      pos.set([s * Math.cos(th), s * Math.sin(th), z], i * 3)
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3))
    return g
  }, [])
  useFrame(({ camera }) => {
    if (ref.current) {
      ref.current.position.copy(camera.position)
      ref.current.scale.setScalar(1e4)
    }
  })
  return (
    <points ref={ref} geometry={geo} renderOrder={-10} frustumCulled={false}>
      <pointsMaterial size={1.1} sizeAttenuation={false} color="#7d7a72" transparent opacity={0.5} depthTest={false} depthWrite={false} />
    </points>
  )
}

function circle(radius: number, n = 160): [number, number, number][] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const a = (i / n) * Math.PI * 2
    return [radius * Math.cos(a), radius * Math.sin(a), 0]
  })
}

/** Distance rings around the Sun in the galactic plane, each shown only at the zoom where it is readable. */
function ScaleRings() {
  const refs = React.useRef<(THREE.Object3D | null)[]>([])
  const labels = React.useRef<(HTMLDivElement | null)[]>([])
  const pts = React.useMemo(() => RINGS.map((r) => circle(r)), [])
  useFrame(({ camera, controls }) => {
    const target = (controls as any)?.target as THREE.Vector3 | undefined
    const d = target ? camera.position.distanceTo(target) : camera.position.length()
    RINGS.forEach((r, i) => {
      const k = r / d
      const op = k < 0.02 || k > 5 ? 0 : Math.min(1, (k - 0.02) / 0.05, (5 - k) / 2)
      const o = refs.current[i] as any
      if (o) {
        o.visible = op > 0.01
        if (o.material) o.material.opacity = 0.5 * op
      }
      const l = labels.current[i]
      if (l) l.style.opacity = String(op)
    })
  })
  return (
    <group>
      {RINGS.map((r, i) => (
        <group key={r}>
          <Line ref={(o) => void (refs.current[i] = o as any)} points={pts[i]} color={AMBER} lineWidth={1} transparent opacity={0.4} depthWrite={false} />
          <Html position={[0, -r, 0]} center zIndexRange={[5, 0]} style={{ pointerEvents: "none" }}>
            <div ref={(el) => void (labels.current[i] = el)} className="atlas-ring-label">
              {fmtLy(r)}
            </div>
          </Html>
        </group>
      ))}
    </group>
  )
}

/** Smooth flight to a goal (target point + distance), cancelled as soon as the user grabs the view. */
/** An object the wheel zooms onto: where it is and how close the camera may get (so that its picture fills the
 *  view instead of the camera flying through it). */
type Aim = { p: [number, number, number]; min: number }
const aimOf = (o: AtlasObject | undefined): Aim | null => (o ? { p: o.xyz, min: photoSize(o) * 1.45 } : null)

function Navigator({ goal, aim, focus, onArrive, onDistance }: {
  goal: Goal | null
  /** the object under the cursor, if any: the wheel zooms onto it */
  aim: Aim | null
  /** the selected object, if any: the wheel zooms onto it when the cursor is on nothing */
  focus: Aim | null
  onArrive: () => void
  onDistance: (d: number) => void
}) {
  const { camera, controls, gl } = useThree()
  const aimRef = React.useRef(aim)
  aimRef.current = aim
  const focusRef = React.useRef(focus)
  focusRef.current = focus
  // the object a zoom gesture started on stays its aim while the gesture goes on, even if the picture slides a
  // few pixels from under the cursor (the layout moved, the hand shook): without this the view drifted away
  const held = React.useRef<{ a: Aim; t: number; x: number; y: number } | null>(null)
  const g = React.useRef<View | null>(null)
  // The wheel. OrbitControls zooms 5 % per wheel event whatever its size and towards the orbit centre: reaching a
  // target took dozens of notches and ended at the Sun, not at the target. Here a notch is ~20 %, a trackpad
  // scrolls smoothly (the step follows the delta), and the zoom goes towards the cursor: onto the object under
  // it when there is one (its real position, so the view converges on it), else onto the point of the view
  // plane under the cursor.
  React.useEffect(() => {
    const el = gl.domElement
    const ray = new THREE.Raycaster()
    const p = new THREE.Vector3()
    const ndc = new THREE.Vector2()
    const wheel = (e: WheelEvent) => {
      const c = controls as any
      if (!c?.target) return
      e.preventDefault()
      if (g.current) {
        g.current = null
        onArrive()
      }
      const delta = Math.max(-360, Math.min(360, e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1)))
      const d = camera.position.distanceTo(c.target)
      const d1 = Math.min(3e9, Math.max(1, d * Math.exp(delta * 0.0019)))
      let s = d1 / d
      const now = performance.now()
      const h = held.current
      const keep = h && now - h.t < 1200 && Math.hypot(e.clientX - h.x, e.clientY - h.y) < 80 ? h.a : null
      const onto = aimRef.current ?? keep ?? focusRef.current
      if (aimRef.current ?? keep) held.current = { a: (aimRef.current ?? keep)!, t: now, x: e.clientX, y: e.clientY }
      if (onto) {
        p.set(...onto.p)
        const dp = camera.position.distanceTo(p)
        if (s < 1) s = dp <= onto.min ? 1 : Math.max(s, onto.min / dp)      // arrived: the picture fills the view
        if (s === 1) return
      } else {
        const r = el.getBoundingClientRect()
        ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
        ray.setFromCamera(ndc, camera)
        p.copy(camera.position).addScaledVector(ray.ray.direction, d)
      }
      // the camera moves along the line to p, the orbit centre keeps its direction at the new distance
      const look = c.target.clone().sub(camera.position)
      camera.position.sub(p).multiplyScalar(s).add(p)
      c.target.copy(camera.position).addScaledVector(look, s)
      c.update()
    }
    el.addEventListener("wheel", wheel, { passive: false })
    return () => el.removeEventListener("wheel", wheel)
  }, [camera, controls, gl, onArrive])
  const last = React.useRef(0)
  React.useEffect(() => {
    const c = controls as any
    if (!goal || !c?.target) {
      g.current = null
      return
    }
    // a goal may leave the centre or the distance as they are (zoom buttons, "home")
    const cur = camera.position.distanceTo(c.target)
    g.current = {
      target: goal.target ?? (c.target.toArray() as [number, number, number]),
      dist: Math.min(3e9, Math.max(1, goal.dist ?? cur * (goal.zoom ?? 1))),
    }
  }, [goal, controls, camera])
  React.useEffect(() => {
    const c = controls as any
    if (!c?.addEventListener) return
    const stop = () => {
      if (g.current) {
        g.current = null
        onArrive()
      }
    }
    c.addEventListener("start", stop)
    return () => c.removeEventListener("start", stop)
  }, [controls, onArrive])
  const v = React.useMemo(() => new THREE.Vector3(), [])
  const off = React.useMemo(() => new THREE.Vector3(), [])
  useFrame((_, dt) => {
    const c = controls as any
    if (!c?.target) return
    const cur = camera.position.distanceTo(c.target)
    if (Math.abs(cur - last.current) > last.current * 0.01) {
      last.current = cur
      onDistance(cur)
    }
    const goalNow = g.current
    if (!goalNow) return
    const t = 1 - Math.exp(-Math.min(dt, 0.1) * 3.2)
    off.copy(camera.position).sub(c.target)
    const d0 = off.length()
    const d1 = Math.exp(Math.log(d0) + (Math.log(goalNow.dist) - Math.log(d0)) * t)
    c.target.lerp(v.set(...goalNow.target), t)
    camera.position.copy(c.target).add(off.normalize().multiplyScalar(d1))
    c.update()
    if (Math.abs(Math.log(d1 / goalNow.dist)) < 0.01 && c.target.distanceTo(v) < goalNow.dist * 0.004) {
      g.current = null
      onArrive()
    }
  })
  return null
}

/** Catalogue objects as dots; hover and click are resolved in screen space (a ray against points of every
 *  size, from 400 to 500 million light years away, would need a threshold per zoom level). */
function Objects({ objects, layers, hovered, selected, onHover, onSelect, onApproach, thumbUrl }: {
  objects: AtlasObject[]
  layers: Layers
  hovered: string | null
  selected: string | null
  onHover: (id: string | null) => void
  onSelect: (id: string | null) => void
  onApproach: (id: string) => void
  thumbUrl: (n: string) => string
}) {
  const { camera, gl, size } = useThree()
  const dim = React.useMemo(() => objects.filter((o) => !o.captured), [objects])
  const geo = React.useMemo(() => {
    const pos = new Float32Array(dim.length * 3)
    const col = new Float32Array(dim.length * 3)
    const c = new THREE.Color()
    dim.forEach((o, i) => {
      pos.set(o.xyz, i * 3)
      c.set(KIND_COLOR[o.kind] ?? KIND_COLOR.other).multiplyScalar(0.75)
      col.set([c.r, c.g, c.b], i * 3)
    })
    const g = new THREE.BufferGeometry()
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3))
    g.setAttribute("color", new THREE.BufferAttribute(col, 3))
    return g
  }, [dim])
  React.useEffect(() => {
    const el = gl.domElement
    const v = new THREE.Vector3()
    let down: [number, number] | null = null
    const nearest = (e: PointerEvent) => {
      const r = el.getBoundingClientRect()
      const mx = e.clientX - r.left
      const my = e.clientY - r.top
      let best: string | null = null
      let bd = 14 * 14
      for (const o of objects) {
        if (!o.captured && !layers.catalogue) continue
        v.set(...o.xyz).project(camera)
        if (v.z > 1 || v.z < -1) continue
        const dx = (v.x * 0.5 + 0.5) * size.width - mx
        const dy = (-v.y * 0.5 + 0.5) * size.height - my
        const d = dx * dx + dy * dy - (o.captured ? 60 : 0) // the photographed ones win when close
        if (d < bd) {
          bd = d
          best = o.id
        }
      }
      return best
    }
    const move = (e: PointerEvent) => {
      const id = e.buttons ? null : nearest(e)
      onHover(id)
      el.style.cursor = id ? "pointer" : "grab"
    }
    const dn = (e: PointerEvent) => void (down = [e.clientX, e.clientY])
    const up = (e: PointerEvent) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) onSelect(nearest(e))
      down = null
    }
    const leave = () => onHover(null)
    const dbl = (e: MouseEvent) => {
      const id = nearest(e as PointerEvent)
      if (id) onApproach(id)
    }
    el.addEventListener("dblclick", dbl)
    el.addEventListener("pointermove", move)
    el.addEventListener("pointerdown", dn)
    el.addEventListener("pointerup", up)
    el.addEventListener("pointerleave", leave)
    return () => {
      el.removeEventListener("dblclick", dbl)
      el.removeEventListener("pointermove", move)
      el.removeEventListener("pointerdown", dn)
      el.removeEventListener("pointerup", up)
      el.removeEventListener("pointerleave", leave)
    }
  }, [objects, camera, gl, size, onHover, onSelect, onApproach, layers.catalogue])

  const shown = objects.filter((o) => o.captured || o.id === hovered || o.id === selected)
  return (
    <group>
      {layers.catalogue && (
        <points geometry={geo} frustumCulled={false}>
          <pointsMaterial size={3.2} sizeAttenuation={false} vertexColors transparent opacity={0.9} depthWrite={false} />
        </points>
      )}
      {objects
        .filter((o) => o.captured && (layers.routes || o.id === selected))
        .map((o) => (
          <Line key={"l" + o.id} points={[[0, 0, 0], o.xyz]} color={RED} lineWidth={o.id === selected ? 1.6 : 0.8} transparent opacity={o.id === selected ? 0.9 : 0.32} depthWrite={false} />
        ))}
      {shown.map((o) => (
        <Html key={o.id} position={o.xyz} center zIndexRange={[o.id === selected ? 40 : o.captured ? 30 : 20, 0]} style={{ pointerEvents: "none" }}>
          <div className={"atlas-mark" + (o.captured ? " got" : "") + (o.captured && o.thumb ? " pic" : "") + (o.id === selected ? " sel" : "") + (o.id === hovered ? " hov" : "") + (layers.labels ? "" : " nolabel")}>
            {o.captured && o.thumb ? <img src={thumbUrl(o.thumb)} alt="" draggable={false} /> : <i />}
            <span>
              {o.name}
              {(o.id === hovered || o.id === selected) && <em>{fmtLy(o.ly)}</em>}
            </span>
          </div>
        </Html>
      ))}
    </group>
  )
}

/** Real size of an object in light years (from its angular size), with a floor so it can always be seen. */
export function sizeLy(o: AtlasObject): number {
  const real = o.size_arcmin ? (o.size_arcmin / 60) * (Math.PI / 180) * o.ly : 0
  return Math.max(real, o.ly * 0.004)
}

function hash(s: string) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

/** Every galaxy of the scene as a small disc of points at its true size and a tilt of its own (not its real
 *  orientation: the catalogue does not carry it in space). Far away they are a dot, close up a galaxy. */
function Galaxies({ objects }: { objects: AtlasObject[] }) {
  const unit = React.useMemo(() => {
    const r = rng(99)
    const N = 520
    const pos = new Float32Array(N * 3)
    for (let i = 0; i < N; i++) {
      const rr = Math.min(1, -0.28 * Math.log(1 - r() * 0.97))
      const th = (i % 2) * Math.PI + Math.log(rr + 0.06) * 3.2 + gauss(r) * 0.35
      pos.set([rr * Math.cos(th), rr * Math.sin(th), gauss(r) * 0.03], i * 3)
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3))
    return g
  }, [])
  const gal = React.useMemo(() => objects.filter((o) => o.kind === "galaxy"), [objects])
  return (
    <group>
      {gal.map((o) => {
        const h = hash(o.id)
        return (
          <points key={o.id} geometry={unit} position={o.xyz} scale={sizeLy(o) / 2} rotation={[((h & 255) / 255) * Math.PI, (((h >> 8) & 255) / 255) * Math.PI, (((h >> 16) & 255) / 255) * Math.PI]}>
            <pointsMaterial size={1.3} sizeAttenuation={false} color={o.captured ? "#dfe6ff" : "#8fa4d8"} transparent opacity={o.captured ? 0.8 : 0.5} depthWrite={false} blending={THREE.AdditiveBlending} />
          </points>
        )
      })}
    </group>
  )
}

/** The photograph of each target, floating at the target at the scale of the object (the field of the picture
 *  is not known here: about twice the catalogue size). Black is transparent (additive), like a slide in the dark. */
function Photos({ objects, thumbUrl }: { objects: AtlasObject[]; thumbUrl: (n: string) => string }) {
  const items = React.useMemo(() => {
    const loader = new THREE.TextureLoader()
    return objects
      .filter((o) => o.captured && o.thumb)
      .map((o) => {
        const tex = loader.load(thumbUrl(o.thumb!))
        tex.colorSpace = THREE.SRGBColorSpace
        return { o, tex }
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [objects.filter((o) => o.captured).map((o) => o.id + o.thumb).join("|")])
  React.useEffect(() => () => items.forEach((i) => i.tex.dispose()), [items])
  // soft edges: the picture fades into space instead of ending in a rectangle
  const fade = React.useMemo(() => {
    const c = document.createElement("canvas")
    c.width = c.height = 128
    const g = c.getContext("2d")!
    const img = g.createImageData(128, 128)
    for (let y = 0; y < 128; y++)
      for (let x = 0; x < 128; x++) {
        const dx = Math.min(x, 127 - x) / 22
        const dy = Math.min(y, 127 - y) / 22
        const a = Math.max(0, Math.min(1, dx)) * Math.max(0, Math.min(1, dy))
        const v = Math.round(255 * a * a * (3 - 2 * a))
        img.data.set([v, v, v, 255], (y * 128 + x) * 4)
      }
    g.putImageData(img, 0, 0)
    return new THREE.CanvasTexture(c)
  }, [])
  return (
    <group>
      {items.map(({ o, tex }) => {
        const s = photoSize(o)
        return (
          <sprite key={o.id} position={o.xyz} scale={[s * 1.5, s, 1]}>
            <spriteMaterial map={tex} alphaMap={fade} transparent opacity={0.95} depthWrite={false} blending={THREE.AdditiveBlending} />
          </sprite>
        )
      })}
    </group>
  )
}
// Height of the photograph in light years. The catalogue sizes are uneven (IC 1396 is listed 14', it spans 3
// degrees; NGC 6357 4'), so pictures were large for some targets and invisible specks for others: every picture is
// at least the field of a typical frame (1.3 degrees at the object's distance), larger when the object is.
export const photoSize = (o: AtlasObject) => Math.max(sizeLy(o) * (o.kind === "galaxy" ? 1.5 : 2), o.ly * 0.023)

function Landmarks() {
  return (
    <group>
      <Html position={[0, 0, 0]} center zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
        <div className="atlas-mark sun">
          <i />
          <span>Sole</span>
        </div>
      </Html>
      <Line points={[[0, 0, 0], GALACTIC_CENTRE]} color={AMBER} lineWidth={0.8} dashed dashSize={600} gapSize={500} transparent opacity={0.45} depthWrite={false} />
      <Html position={GALACTIC_CENTRE} center zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
        <div className="atlas-mark centre">
          <i />
          <span>
            Centro galattico<em>26,7 mila al</em>
          </span>
        </div>
      </Html>
      <Line points={circle(52000).map(([x, y, z]) => [x + GALACTIC_CENTRE[0], y, z] as [number, number, number])} color={PAPER} lineWidth={0.6} transparent opacity={0.14} depthWrite={false} />
    </group>
  )
}

export function Scene(props: {
  objects: AtlasObject[]
  active: boolean
  goal: Goal | null
  hovered: string | null
  selected: string | null
  onHover: (id: string | null) => void
  onSelect: (id: string | null) => void
  onApproach: (id: string) => void
  onArrive: () => void
  onDistance: (d: number) => void
  start: View
  layers: Layers
  thumbUrl: (n: string) => string
}) {
  const aim = React.useMemo(() => aimOf(props.objects.find((o) => o.id === props.hovered)), [props.objects, props.hovered])
  const focus = React.useMemo(() => aimOf(props.objects.find((o) => o.id === props.selected)), [props.objects, props.selected])
  const startPos = React.useMemo(() => {
    const d = props.start.dist
    const [tx, ty, tz] = props.start.target
    return [tx - d * 0.55, ty - d * 0.62, tz + d * 0.56] as [number, number, number]
  }, [props.start])
  return (
    <Canvas
      dpr={[1, 2]}
      frameloop={props.active ? "always" : "never"}
      gl={{ antialias: true, logarithmicDepthBuffer: true, powerPreference: "high-performance" }}
      camera={{ position: startPos, up: [0, 0, 1], near: 1, far: 5e12, fov: 42 }}
      onCreated={({ gl }) => gl.setClearColor("#050505")}
    >
      <OrbitControls makeDefault target={props.start.target} enableDamping dampingFactor={0.09} minDistance={1} maxDistance={3e9} enableZoom={false} rotateSpeed={0.6} panSpeed={0.8} />
      <Navigator goal={props.goal} aim={aim} focus={focus} onArrive={props.onArrive} onDistance={props.onDistance} />
      <Backdrop />
      {props.layers.galaxy && <MilkyWay />}
      {props.layers.rings && <ScaleRings />}
      <Landmarks />
      {props.layers.galaxy && <Galaxies objects={props.objects} />}
      {props.layers.photos && <Photos objects={props.objects} thumbUrl={props.thumbUrl} />}
      <Objects objects={props.objects} layers={props.layers} hovered={props.hovered} selected={props.selected} onHover={props.onHover} onSelect={props.onSelect} onApproach={props.onApproach} thumbUrl={props.thumbUrl} />
    </Canvas>
  )
}

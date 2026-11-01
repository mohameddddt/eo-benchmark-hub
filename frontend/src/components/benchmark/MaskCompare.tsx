import { useRef, useState } from "react";

/** Before/after slider: the photo underneath, two masks blended on top, split
 *  by a draggable handle. Keyboard: ←/→ move the handle. */
export default function MaskCompare({ image, left, right, opacity = 0.62 }: {
  image: string; left: string; right: string; opacity?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50);
  const [alpha, setAlpha] = useState(opacity);
  const move = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (r) setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  };
  return (
    <div>
      <div className="compare-frame" ref={ref}
           onPointerDown={(e) => { (e.target as HTMLElement).setPointerCapture?.(e.pointerId); move(e.clientX); }}
           onPointerMove={(e) => { if (e.buttons) move(e.clientX); }}>
        <img className="base" src={image} alt="Aerial tile" />
        <img src={right} alt="Ground-truth mask" style={{ opacity: alpha, clipPath: `inset(0 0 0 ${pos}%)` }} />
        <img src={left} alt="Predicted mask" style={{ opacity: alpha, clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
        <div className="compare-handle" style={{ left: `${pos}%` }} role="slider" tabIndex={0} aria-label="Comparison position"
             aria-valuenow={Math.round(pos)} aria-valuemin={0} aria-valuemax={100}
             onKeyDown={(e) => {
               if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 5));
               if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 5));
             }} />
        <span className="badge neutral" style={{ position: "absolute", top: 10, left: 10 }}>Prediction</span>
        <span className="badge neutral" style={{ position: "absolute", top: 10, right: 10 }}>Ground truth</span>
      </div>
      <label className="row xs muted" style={{ marginTop: 10, gap: 10 }}>
        Mask opacity
        <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(+e.target.value)} style={{ width: 140, padding: 0 }} />
      </label>
    </div>
  );
}

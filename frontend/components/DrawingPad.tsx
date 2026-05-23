"use client"

import { useRef, useEffect, forwardRef, useImperativeHandle } from "react"

export interface DrawingPadHandle {
  getDataUrl: () => string | null
  clear: () => void
}

const DrawingPad = forwardRef<DrawingPadHandle>(function DrawingPad(_, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawing = useRef(false)
  const lastPos = useRef<{ x: number; y: number } | null>(null)
  const hasDrawn = useRef(false)

  function initCtx(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d")!
    ctx.fillStyle = "#ffffff"
    ctx.fillRect(0, 0, 280, 280)
    ctx.strokeStyle = "#000000"
    ctx.lineWidth = 20
    ctx.lineCap = "round"
    ctx.lineJoin = "round"
  }

  useEffect(() => {
    if (canvasRef.current) initCtx(canvasRef.current)
  }, [])

  useImperativeHandle(ref, () => ({
    clear() {
      hasDrawn.current = false
      if (canvasRef.current) initCtx(canvasRef.current)
    },
    getDataUrl() {
      if (!hasDrawn.current || !canvasRef.current) return null
      const off = document.createElement("canvas")
      off.width = 28
      off.height = 28
      const ctx = off.getContext("2d")!
      ctx.drawImage(canvasRef.current, 0, 0, 28, 28)
      const id = ctx.getImageData(0, 0, 28, 28)
      const d = id.data
      for (let i = 0; i < d.length; i += 4) {
        // Luminance-weighted grayscale, then invert: white bg → black, black stroke → white
        const g = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2])
        d[i] = d[i + 1] = d[i + 2] = 255 - g
      }
      ctx.putImageData(id, 0, 0)
      return off.toDataURL("image/png")
    },
  }))

  function getPos(e: React.MouseEvent | React.TouchEvent) {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const sx = canvas.width / rect.width
    const sy = canvas.height / rect.height
    if ("touches" in e.nativeEvent) {
      const t = (e as React.TouchEvent).touches[0]
      return { x: (t.clientX - rect.left) * sx, y: (t.clientY - rect.top) * sy }
    }
    const m = e as React.MouseEvent
    return { x: (m.clientX - rect.left) * sx, y: (m.clientY - rect.top) * sy }
  }

  function onStart(e: React.MouseEvent | React.TouchEvent) {
    drawing.current = true
    hasDrawn.current = true
    lastPos.current = getPos(e)
  }

  function onMove(e: React.MouseEvent | React.TouchEvent) {
    if (!drawing.current) return
    const pos = getPos(e)
    const ctx = canvasRef.current!.getContext("2d")!
    ctx.beginPath()
    ctx.moveTo(lastPos.current!.x, lastPos.current!.y)
    ctx.lineTo(pos.x, pos.y)
    ctx.stroke()
    lastPos.current = pos
  }

  function onEnd() {
    drawing.current = false
    lastPos.current = null
  }

  return (
    <canvas
      ref={canvasRef}
      width={280}
      height={280}
      className="border-2 border-gray-300 rounded-md cursor-crosshair bg-white"
      style={{ touchAction: "none" }}
      onMouseDown={onStart}
      onMouseMove={onMove}
      onMouseUp={onEnd}
      onMouseLeave={onEnd}
      onTouchStart={onStart}
      onTouchMove={onMove}
      onTouchEnd={onEnd}
    />
  )
})

export default DrawingPad

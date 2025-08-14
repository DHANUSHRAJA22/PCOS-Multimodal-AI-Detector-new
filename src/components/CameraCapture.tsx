// src/components/CameraCapture.tsx
import { useEffect, useRef, useState, useCallback } from 'react'
import { X, Camera, RefreshCcw, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

type Props = {
  /** Controls the visibility of the camera modal */
  open: boolean
  /** Called when the modal should be opened/closed */
  onOpenChange: (open: boolean) => void
  /** Called with a captured file (JPEG) when the user hits “Capture” */
  onCapture: (file: File) => void
  /** Which camera to use by default */
  facingMode?: 'user' | 'environment'
  /** Optional aspect ratio (e.g., 4/3, 16/9). If omitted, uses video’s natural size */
  aspectRatio?: number
  /** Optional CSS class for the modal container */
  className?: string
}

export function CameraCapture({
  open,
  onOpenChange,
  onCapture,
  facingMode: initialFacing = 'environment',
  aspectRatio,
  className,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [facing, setFacing] = useState<'user' | 'environment'>(initialFacing)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
  }, [])

  const startStream = useCallback(async (mode: 'user' | 'environment') => {
    setError(null)
    setStarting(true)
    try {
      // try exact facing mode first, then fall back
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        // iOS needs playsInline + muted + autoplay true
        await videoRef.current.play().catch(() => {})
      }
    } catch (e: any) {
      setError(
        e?.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow access to your camera.'
          : 'Unable to access the camera on this device.'
      )
    } finally {
      setStarting(false)
    }
  }, [])

  // open/close lifecycle
  useEffect(() => {
    if (open) {
      startStream(facing)
    } else {
      stopStream()
    }
    return () => stopStream()
  }, [open, startStream, stopStream, facing])

  const toggleFacing = useCallback(() => {
    const next = facing === 'user' ? 'environment' : 'user'
    setFacing(next)
    // restart stream on next render
    stopStream()
    startStream(next)
  }, [facing, startStream, stopStream])

  const doCapture = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const vw = video.videoWidth || 1280
    const vh = video.videoHeight || 720

    // compute target size honoring optional aspect ratio
    let cw = vw
    let ch = vh
    if (aspectRatio && aspectRatio > 0) {
      const current = vw / vh
      if (current > aspectRatio) {
        // too wide → pillarbox
        ch = vh
        cw = Math.round(vh * aspectRatio)
      } else {
        // too tall → letterbox
        cw = vw
        ch = Math.round(vw / aspectRatio)
      }
    }

    canvas.width = cw
    canvas.height = ch

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // center-crop if we enforced aspect
    const sx = Math.floor((vw - cw) / 2)
    const sy = Math.floor((vh - ch) / 2)
    ctx.drawImage(video, sx, sy, cw, ch, 0, 0, cw, ch)

    await new Promise<void>((resolve) => {
      canvas.toBlob(
        (blob) => {
          if (!blob) return
          const file = new File([blob], `capture_${Date.now()}.jpg`, { type: 'image/jpeg' })
          onCapture(file)
          onOpenChange(false)
          resolve()
        },
        'image/jpeg',
        0.92
      )
    })
  }, [onCapture, onOpenChange, aspectRatio])

  if (!open) return null

  return (
    <div
      className={cn(
        'fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm',
        className
      )}
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-2xl mx-auto rounded-xl overflow-hidden bg-white shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <div className="font-semibold">Camera</div>
          <Button variant="ghost" size="icon" onClick={() => onOpenChange(false)} aria-label="Close">
            <X className="h-5 w-5" />
          </Button>
        </div>

        <div className="relative bg-black">
          <video
            ref={videoRef}
            className="w-full h-[60vh] object-contain bg-black"
            playsInline
            muted
            autoPlay
          />
          {starting && (
            <div className="absolute inset-0 grid place-items-center bg-black/50 text-white text-sm">
              Starting camera…
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="m-6 max-w-sm rounded-lg bg-white p-4 shadow">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-5 w-5 text-orange-600 mt-0.5" />
                  <div>
                    <div className="font-medium mb-1">Camera error</div>
                    <div className="text-sm text-slate-600">{error}</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 p-4">
          <div className="flex gap-2">
            <Button variant="outline" onClick={toggleFacing} title="Switch camera">
              <RefreshCcw className="h-4 w-4 mr-2" />
              Switch
            </Button>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={doCapture} disabled={!!error || starting}>
              <Camera className="h-4 w-4 mr-2" />
              Capture
            </Button>
          </div>
        </div>
      </div>

      {/* offscreen canvas used for capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

export default CameraCapture

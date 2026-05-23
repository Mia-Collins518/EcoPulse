"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import DrawingPad, { type DrawingPadHandle } from "@/components/DrawingPad"

const InferencePage = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [actual, setActual] = useState<number | null>(null)
  const [predicted, setPredicted] = useState<number | null>(null)
  const [image, setImage] = useState<string | null>(null)
  const [mode, setMode] = useState<string | null>(null)
  const [customTab, setCustomTab] = useState<"draw" | "upload">("draw")
  const [file, setFile] = useState<File | null>(null)
  // uploadPreview: the user's original file, never replaced by inference results
  const [uploadPreview, setUploadPreview] = useState<string | null>(null)
  // processedImage: the 28×28 image the model actually saw, set after inference
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const padRef = useRef<DrawingPadHandle>(null)

  function resetResults() {
    setActual(null)
    setPredicted(null)
    setImage(null)
    setError(null)
    setLoading(false)
    setProcessedImage(null)
  }

  async function fetchRandom() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("http://localhost:8000/api/random-inference")
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const body = await res.json()
      setActual(body.actual)
      setPredicted(body.predicted)
      setImage(body.image)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  async function runCustomInference(dataUrl: string) {
    setLoading(true)
    setError(null)
    setPredicted(null)
    setProcessedImage(null)
    try {
      const res = await fetch("http://localhost:8000/api/custom-inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const body = await res.json()
      setPredicted(body.predicted)
      if (body.image) setProcessedImage(body.image)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null
    setFile(f)
    setPredicted(null)
    setError(null)
    setProcessedImage(null)
    if (f) {
      const reader = new FileReader()
      reader.onload = () => setUploadPreview(reader.result as string)
      reader.readAsDataURL(f)
    } else {
      setUploadPreview(null)
    }
  }

  function handleDrawSubmit() {
    const dataUrl = padRef.current?.getDataUrl()
    if (!dataUrl) {
      setError("Please draw a digit first.")
      return
    }
    // Show the exported canvas immediately so the user sees what's being sent
    setProcessedImage(dataUrl)
    runCustomInference(dataUrl)
  }

  return (
    <div className="flex flex-col items-center justify-center w-screen bg-background text-center p-8 space-y-4">
      <header className="mb-6">
        <h1 className="text-5xl font-bold">Run Inference</h1>
        <p className="text-xl p-3">
          In machine learning, inference is the process of using a trained model to make predictions on new, unseen data.<br />
          Use this page to run a random MNIST digit through the model, draw your own, or upload an image.
        </p>
      </header>

      {/* Mode selection */}
      <div className="flex gap-8 mb-8">
        <Button
          onClick={() => {
            resetResults()
            setUploadPreview(null)
            setFile(null)
            setMode("custom")
          }}
          disabled={loading || mode === "custom"}
          className="w-[200] h-[100] text-xl bg-brand-light"
        >
          Custom Inference
        </Button>
        <Button
          onClick={() => {
            resetResults()
            setUploadPreview(null)
            setFile(null)
            setMode("random")
            fetchRandom()
          }}
          disabled={loading}
          className="w-[200] h-[100] text-xl bg-brand-light"
        >
          Random Inference
        </Button>
      </div>

      {mode === null && (
        <p className="text-gray-500">Select one of the buttons above to begin.</p>
      )}

      {/* Custom inference */}
      {mode === "custom" && (
        <div className="flex flex-col items-center space-y-6">
          {/* Draw / Upload tab bar */}
          <div className="flex border-b border-gray-300">
            <button
              onClick={() => {
                setCustomTab("draw")
                resetResults()
                setUploadPreview(null)
                setFile(null)
                padRef.current?.clear()
              }}
              className={`px-8 py-2 text-lg font-medium transition-colors ${
                customTab === "draw"
                  ? "border-b-2 border-brand-medium text-brand-medium"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Draw
            </button>
            <button
              onClick={() => {
                setCustomTab("upload")
                resetResults()
                setUploadPreview(null)
                setFile(null)
              }}
              className={`px-8 py-2 text-lg font-medium transition-colors ${
                customTab === "upload"
                  ? "border-b-2 border-brand-medium text-brand-medium"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Upload
            </button>
          </div>

          {/* Draw tab */}
          {customTab === "draw" && (
            <div className="flex flex-col items-center space-y-4">
              <p className="text-gray-500 text-sm">Draw a digit (0–9) on the canvas below</p>
              <DrawingPad ref={padRef} />
              <div className="flex gap-4">
                <Button
                  variant="secondary"
                  onClick={() => { padRef.current?.clear(); resetResults() }}
                  className="w-32 h-12 text-lg bg-brand-light text-white"
                >
                  Clear
                </Button>
                <Button
                  onClick={handleDrawSubmit}
                  disabled={loading}
                  className="w-44 h-12 text-lg bg-brand-light"
                >
                  {loading ? "Running…" : "Run Inference"}
                </Button>
              </div>
            </div>
          )}

          {/* Upload tab */}
          {customTab === "upload" && (
            <div className="flex flex-col items-center space-y-4">
              <p className="text-gray-500 text-sm">Upload an image of a handwritten digit</p>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="border p-2 rounded"
              />
              {/* Original upload — always kept, never replaced by inference results */}
              {uploadPreview && (
                <img
                  src={uploadPreview}
                  alt="Upload preview"
                  className="w-64 h-64 border object-contain rounded"
                />
              )}
              <Button
                onClick={() => uploadPreview && runCustomInference(uploadPreview)}
                disabled={loading || !file}
                className="w-44 h-12 text-xl bg-brand-light"
              >
                {loading ? "Running…" : "Run Inference"}
              </Button>
            </div>
          )}

          {/* Shared results for both custom tabs */}
          {error && <p className="text-red-500">{error}</p>}

          {!loading && predicted !== null && (
            <div className="flex flex-col items-center space-y-3">
              {processedImage && (
                <div className="flex flex-col items-center gap-1">
                  <p className="text-sm text-gray-500">What the model sees (28×28):</p>
                  <img
                    src={processedImage}
                    alt="Processed digit"
                    width={112}
                    height={112}
                    className="border"
                    style={{ imageRendering: "pixelated" }}
                  />
                </div>
              )}
              <p className="text-2xl">
                Predicted Digit: <strong>{predicted}</strong>
              </p>
            </div>
          )}

          <Button
            variant="secondary"
            onClick={() => {
              setMode(null)
              resetResults()
              setUploadPreview(null)
              setFile(null)
              padRef.current?.clear()
            }}
            className="w-40 h-17 text-2xl font-semibold bg-brand-light text-white"
          >
            Reset
          </Button>
        </div>
      )}

      {/* Random inference */}
      {mode === "random" && (
        <div className="space-y-4 flex flex-col items-center">
          {loading && <p className="text-gray-500">Thinking… please wait</p>}

          {/* Show errors so the button doesn't appear to do nothing */}
          {!loading && error && (
            <>
              <p className="text-red-500">{error}</p>
              <Button
                variant="secondary"
                onClick={() => { setMode(null); resetResults() }}
                className="w-40 h-17 text-2xl font-semibold bg-brand-light text-white"
              >
                Reset
              </Button>
            </>
          )}

          {!loading && actual !== null && predicted !== null && image && (
            <>
              <img
                src={image}
                alt={`Digit ${actual}`}
                className="w-108 h-108 border object-contain"
              />
              <p className="text-2xl">Actual: <strong>{actual}</strong></p>
              <p className="text-2xl">Predicted: <strong>{predicted}</strong></p>
              <Button
                variant="secondary"
                onClick={() => { setMode(null); resetResults() }}
                className="w-40 h-17 text-2xl font-semibold bg-brand-light text-white"
              >
                Reset
              </Button>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default InferencePage

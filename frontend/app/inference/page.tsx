"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"

const  InferencePage = () => {
    // loading / error flags
    const [loading, setLoading]   = useState(false)
    const [error,   setError]     = useState<string | null>(null)

    // the two bits of data we want: actual label & model’s guess
    const [actual,    setActual]    = useState<number | null>(null)
    const [predicted, setPredicted] = useState<number | null>(null)

    // the Base64-encoded PNG data URL for the image
    const [image,     setImage]     = useState<string | null>(null)

    // Conditional sub-UI state
    const [mode, setMode] = useState<string | null>(null)

    // file selected by the user
    const [file, setFile] = useState<File | null>(null)
    // preview Data URL so you can show the image immediately
    const [preview, setPreview] = useState<string | null>(null)

    async function fetchRandom() {
        setLoading(true)
        setError(null)

        try{
            const res = await fetch("http://localhost:8000/api/random-inference")
            if (!res.ok) throw new Error(`Server returned ${res.status}`)

            const body = await res.json()
            setActual(body.actual)       
            setPredicted(body.predicted)
            setImage(body.image)
        } catch (err: any) {
            setError (err.message || "Unknown error")
        } finally {
            setLoading(false)
        }
    }

    // Handles file slection and preview
    function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
        const f = e.target.files?.[0] ?? null // grab file object that the user inputs 
        setFile(f)
        // Creates a file reader to upload the image for display
        if (f) {
            const reader = new FileReader()
            reader.onload = () => setPreview(reader.result as string)
            reader.readAsDataURL(f)
        } else {
            setPreview(null)
        }
    }

    // POSTs the image to the API and recives the prediction
    async function fetchCustom() {
        if (!file || !preview) return
            setLoading(true)
            setError(null)
        try {
            const res = await fetch("http://localhost:8000/api/custom-inference", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: preview })
            })
            if (!res.ok) throw new Error(`Server returned ${res.status}`)
                const body = await res.json()
                // assume backend returns { predicted: number, image?: string }
                setPredicted(body.predicted)
            // you can also update preview if backend returns a cleaned-up image
            if (body.image) setPreview(body.image)
        } catch (err: any) {
            setError(err.message || "Unknown error")
        } finally {
            setLoading(false)
        }
    }

    // flex items-center justify-between px-4 w-full
    //max-w-3xl mx-auto p-8
    // flex flex-col items-center justify-center w-screen bg-background text-center p-8 space-y-4
    return (
        <div className="flex flex-col items-center justify-center w-screen bg-background text-center p-8 space-y-4">
            <header className="mb-6">
                <h1 className="text-5xl font-bold">Run Inference</h1>
                <p className="text-xl p-3">
                In machine learning, inference is the process of using a trained model to make predictions on new, unseen data. <br></br>
                Use this page to either run a random MNIST digit through the model, or upload your own.
                </p>
            </header>

            {/* Selection Buttons */}
            <div className="flex gap-8 mb-8">
                <Button 
                    onClick={() => {
                        // clear any previous results
                        setActual(null);
                        setPredicted(null);
                        setImage(null);
                        setError(null);
                        setLoading(false);
                        // switch into "custom" mode
                        setMode("custom");
                    }}
                    disabled={loading || mode === "custom"} 
                    className="w-[200] h-[100] text-xl bg-brand-light"
                >
                    Custom Inference
                </Button>


                <Button 
                    onClick={() => {
                        // clear any previous results
                        setActual(null);
                        setPredicted(null);
                        setImage(null);
                        setError(null);
                        setLoading(false);
                        // switch into "random" mode
                        setMode("random");
                        fetchRandom()
                    }}
                    disabled={loading} 
                    className="w-[200] h-[100] text-xl bg-brand-light"
                >
                    {/* {loading ? "Thinking…" : "Random Inference"}*/} 
                    Random Inference
                </Button>

            </div>

            {/* If no mode is selected, then the prompt is shown */}
            {mode === null && (
            <p className="text-gray-500">
                Select one of the buttons above to begin.
            </p>
            )}

            {/* If in "custom" mode, then stuff needs to be added */}
            {mode === "custom" && (
            <p className="text-gray-500">
                {mode === "custom" && (
                <div className="flex flex-col items-center space-y-4">
                    <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="border p-2"
                    />
                    {preview && (
                    <img
                        src={preview}
                        alt="Your upload preview"
                        className="w-108 h-108 border object-contain"
                    />
                    )}
                
                    <Button
                    onClick={fetchCustom}
                    disabled={loading || !file}
                    className="w-40 h-12 text-xl bg-brand-light"
                    >
              
                    {loading ? "Running…" : "Run Inference"}
                    </Button>
                    {error && <p className="text-red-500">{error}</p>}
                    {!loading && predicted !== null && (
                    <p className="text-2xl">
                        Predicted Digit: <strong>{predicted}</strong>
                    </p>
                    )}

                    <Button
                        variant="secondary"
                        onClick={() => {
                            // “Reset” resets everything and returns to mode=null
                            setMode(null);
                            setActual(null);
                            setPredicted(null);
                            setImage(null);
                            setError(null);
                            setLoading(false);
                        }}
                        className="w-40 h-17 text-2xl font-semibold bg-brand-light text-white"
                    >
                        Reset
                    </Button>
                </div>
                )}
            </p>
            )}

            {/* If in “random” mode, show loading / result display */}
            {mode === "random" && (
            <div className="space-y-4 flex flex-col items-center">
                {loading && <p className="text-gray-500">Thinking… please wait</p>}

                {!loading && actual !== null && predicted !== null && image && (
                <>
                    <img
                    src={image}
                    alt={`Digit ${actual}`}
                    className="w-108 h-108 border object-contain"
                    />
                    <p className="text-2xl">
                    Actual: <strong>{actual}</strong>
                    </p>
                    <p className="text-2xl">
                    Predicted: <strong>{predicted}</strong>
                    </p>
                    
                    <Button
                        variant="secondary"
                        onClick={() => {
                            // “Reset” resets everything and returns to mode=null
                            setMode(null);
                            setActual(null);
                            setPredicted(null);
                            setImage(null);
                            setError(null);
                            setLoading(false);
                        }}
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


export default InferencePage;
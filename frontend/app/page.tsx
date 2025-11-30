import { Button } from "@/components/ui/button"

export default function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen w-screen bg-background px-4 text-center">
        {/* Logo and Title */}
        <div className="flex-inital mb-8">
          <h1 className="text-9xl font-bold mt-4 text-brand-medium ">EcoPulse</h1>
          <h2 className="mt-2 text-6xl font-bold text-brand-medium">Sustainable AI Technologies </h2>

          <img src="/logo.png" alt="EcoPulse Logo" className=" w-[1000] h-[550] mx-auto" />
          
        </div>

      {/* Navigation Buttons */}
      <div className="flex gap-8">
        <Button asChild className="w-[300] h-[150] text-2xl bg-brand-light"><a href="/inference">Run Inference</a></Button>
        <Button asChild className="w-[300] h-[150] text-2xl bg-brand-light"><a href="/metrics">Metrics</a></Button>
        <Button asChild className="w-[300] h-[150] text-2xl bg-brand-light"><a href="/about">About</a></Button>
      </div>
    </div>
  )
}
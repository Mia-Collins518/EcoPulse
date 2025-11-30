"use client"

import { useEffect, useState } from "react"
import { Line } from "react-chartjs-2"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js"

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

import { Settings } from "lucide-react"

// Define a TypeScript type for all metrics data
type MetricsData = Record<string, { step: number; value: number }[]>

// React component 
export default function MetricsPage() {
  // useState to define the current state, and the method to update the state
  const [projects,     setProjects]    = useState<string[]>([""]) // list of project names ***HARDCODED RN***
  const [tags,         setTags]        = useState<string[]>([]) // list of metric names
  const [selProject,   setProject]     = useState<string>("test_2") // currently selected project, default in ""
  const [metricsData,  setMetricsData] = useState<MetricsData>({}) // holds arrays for each tag

  const lossData = metricsData["loss_train"]     || [];
  const accData  = metricsData["accuracy_train"] || [];

  // keys in metricsData besides "loss_train" and "accuracy_train"
  const extraTags = [
    "parameters_model",
    "flops_model",
    "energy_train_kwh",
    "time_train"
  ];

  // 1) Number of epochs = the highest “step” in lossData (or accData)
  const numEpochs =
    lossData.length > 0
      ? lossData[lossData.length - 1].step
      : 0;

  // 2) Final accuracy = last entry in accData (or “—” if no data)
  const finalAccuracy =
    accData.length > 0
      ? accData[accData.length - 1].value
      : "—";

  // ── src/app/metrics/page.tsx ──
  const card_info: Record<string, { title: string; description: string }> = {
    "parameters_model": {
      title: "Model Parameters",
      description: "Total number of trainable parameters (internal knobs and switches) in the network."
    },
    "flops_model": {
      title: "Floating-Point Operations",
      description:
        "An estimate of how many multiply/add operations the model performs every time it tries to recongize a digit."
    },
    "energy_train_kwh": {
      title: "Estimated Training Energy",
      description:
        "Approximate energy consumption (kWh) for training the model over all epochs."
    },
    "time_train": {
      title: "Training Time",
      description:
        "The total amount of time (in minutes) it took to train this model."
    },
    "Epochs": {
      title: "Epochs Completed",
      description:
        "The total number of training rounds that were run. Each round, the model references the full training dataset."
    },
    "Final Accuracy": {
      title: "Final Accuracy (%)",
      description:
        "Accuracy achieved at the last epoch of training."
    }
  };

  // 1. On initial rendering, fetch your list of projects 
  useEffect(() => {
    fetch("http://localhost:8000/api/projects")
      .then(res => res.json())
      .then(json => {setProjects(json.projects)})  // json.projects is the array returned by your FastAPI route
  }, []) 
  // the empty `[]` means “run this effect once, when the component first appears.”

  // 2. When the user picks a project, fetch its tags
  useEffect(() => {
    if (!selProject) return  // nothing selected yet
    fetch(`http://localhost:8000/api/${selProject}/tags`)
      .then(res => res.json())
      .then(json => setTags(json.tags))
      .catch(console.error)
  }, [selProject])  
  // this effect re-runs *only* when selProject changes

  // 3. Once tags arrive, fetch all the metrics data in parallel
  useEffect(() => {
    if (tags.length === 0) return

    Promise.all(
      tags.map(tag =>
        fetch(`http://localhost:8000/api/${selProject}/${tag}`)
          .then(r => r.json())
          .then(j => ({ tag, data: j.data }))
      )
    )
      .then(results => {
        const map: MetricsData = {}
        results.forEach(({ tag, data }) => {
          map[tag] = data
        })
        setMetricsData(map)
      })
      .catch(console.error)
  }, [selProject, tags])

  // Render the charts with react-chartjs-2
  return (
    // Title and description at the top
    <div className="p-8 space-y-4">
        <header className="flex items-center justify-center px-4 w-full">
            <h1 className="text-5xl font-bold">Training Metrics</h1>
        </header>
        
        <div className="flex items-center justify-between px-50"> 
            <div className="px-10 text-xl"> 
                <p> This page shows how the digit-recognition model learns over time. The cards share facts about the model, while the charts below that illustrate how it's predictions got better as it sees more examples. We can also compare different training sessions. </p>
            </div>

            <Dialog>
                <DialogTrigger asChild>
                <Button variant="outline" className=" w-16 h-14">
                    <Settings className="size-8"/>
                </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                    <DialogTitle>Settings</DialogTitle>
                    <DialogDescription>
                        Adjust the metrics display.
                    </DialogDescription>
                    </DialogHeader>
                    <div className="flex items-center space-x-2">
                    <div className="grid flex-1 gap-2">

                        <label>
                            Project:
                            <select
                            value={selProject}
                            onChange={e => setProject(e.target.value)}
                            className="ml-2 border rounded px-2"
                            >
                            {projects.map(p => (
                                <option key={p} value={p}>{p}</option>
                            ))}
                            </select>
                        </label>
            
                    </div>
                    </div>
                </DialogContent>
            </Dialog>
        </div>

      <div className="flex flex-col items-center gap-12">
        {/* ── Static “cards” ── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mt-8">
          {extraTags.map((tag) => {
            const info = card_info[tag];
            const rawData = metricsData[tag] ?? [];

            // Decide which element to pull: 
            // • For energy/params/flops you use the 0th element
            // • For time, grab the last element
            const value: number | null = (() => {
              if (rawData.length === 0) return null;

              if (tag === "time_train") {
                // “time_train” holds an array of { step, value } for each epoch.
                // We want the very last one: rawData[rawData.length - 1].value
                return rawData[rawData.length - 1].value;
              } else {
                // Everything else we logged at epoch 0, so index 0 is fine
                return rawData[0].value;
              }
            })();

            // If this is the energy‐kWh tag, format in exponential form
            const display = (() => {
              if (value === null) return "—";

              if (tag === "energy_train_kwh") {
              // multiply single‐step estimate by numEpochs to get total‐training energy
              const totalEnergy = (value as number) * numEpochs;
              return totalEnergy.toExponential(3) + " kWh";
              } else if (tag === "time_train") {
                // value is in seconds; convert to minutes
                const minutes = (value as number) / 60;
                // e.g. show two decimal places
                return minutes.toFixed(2) + " min";
              }

              // default for params and flops:
              return (value as number).toLocaleString();
            })();

            return (
              <div
                key={tag}
                className="bg-white p-4 rounded-lg shadow-md w-[350px] h-[175px]"
              >
                <p className="text-xl text-gray-500 mb-2">{info.title}</p>
                <p className="text-3xl font-semibold mb-2">{display}</p>
                <p className="text-base text-gray-600">{info.description}</p>
              </div>
            );
          })}

          {/* Epochs Completed card */}
          <div className="bg-white p-4 rounded-lg shadow-md w-[350px] h-[175px]">
            <p className="text-xl text-gray-500 mb-2">{card_info["Epochs"].title}</p>
            <p className="text-3xl font-semibold mb-2">{numEpochs.toLocaleString()}</p>
            <p className="text-base text-gray-600">{card_info["Epochs"].description}</p>
          </div>

          {/* Final Accuracy card */}
          <div className="bg-white p-4 rounded-lg shadow-md w-[350px] h-[175px]">
            <p className="text-xl text-gray-500 mb-2">{card_info["Final Accuracy"].title}</p>
            <p className="text-3xl font-semibold mb-2">
              {typeof finalAccuracy === "number"
                ? finalAccuracy.toLocaleString(undefined, { maximumFractionDigits: 2 })
                : "—"}
            </p>
            <p className="text-base text-gray-600">{card_info["Final Accuracy"].description}</p>
          </div>
        </div>

        {/* Accuracy chart + description */}
        <div className="flex flex-col md:flex-row items-start md:items-center gap-6 w-full max-w-[1000px]">
          <div className="flex-1">
            <h2 className="text-2xl font-semibold mb-2">Accuracy over Epochs</h2>
            <p className="text-gray-600 mb-4">
              As training proceeds, “Accuracy” should rise. Ideally, you see an upward trend toward 100%.
            </p>
            <Line
              data={{
                labels: accData.map((d) => d.step),
                datasets: [
                  {
                    label: "Accuracy (%)",
                    data: accData.map((d) => d.value),
                    borderColor: "#10b981",
                    fill: false,
                    tension: 0.3,
                  },
                ],
              }}
              options={{
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                  x: { title: { display: true, text: "Epoch" } },
                  y: { title: { display: true, text: "Accuracy (%)" } },
                },
              }}
              className="w-full h-64"
            />
          </div>
        </div>

        {/* 2A) Loss chart + description */}
        <div className="flex flex-col md:flex-row items-start md:items-center gap-6 w-full max-w-[1000px]">
          <div className="flex-1">
            <h2 className="text-2xl font-semibold mb-2">Loss over Epochs</h2>
            <p className="text-gray-600 mb-4">
              As training proceeds, ‘Loss’ should decrease. A smoothly descending curve shows the model is steadily improving. 
            </p>
            <Line
              data={{
                labels: lossData.map((d) => d.step),
                datasets: [
                  {
                    label: "Loss",
                    data: lossData.map((d) => d.value),
                    borderColor: "#3b82f6",
                    fill: false,
                    tension: 0.3,
                  },
                ],
              }}
              options={{
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                  x: { title: { display: true, text: "Epoch" } },
                  y: { title: { display: true, text: "Loss" } },
                },
              }}
              className="w-full h-64"
            />
          </div>
        </div>
      </div>
    </div>
  )
}
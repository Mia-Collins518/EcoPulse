"use client";

import { useEffect, useState } from "react";

export default function AboutPage() {
  // ─── Compute “homes equivalent” ─────────────────────────────────────────────
  // Total GPT-4 training energy in kWh (≈57 million kWh)
  const gpt4TrainingKWh = 57_000_000;
  // Average annual energy per U.S. home in kWh
  const oneHomeKWh = 14_380;
  // Calculate how many homes that energy could power for one year
  const gpt4HomesEquivalent = gpt4TrainingKWh / oneHomeKWh; // ≈3963 homes

  // ─── Generate an array of ~3,963 “house” entries ───────────────────────────
  const totalHouses = Math.round(gpt4HomesEquivalent);
  const housesArray = Array.from({ length: totalHouses }, (_, i) => i);

  return (
    <div>
      {/* Reduce horizontal padding from p-8 to px-6, use larger max-width */}
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        {/* Page title */}
        <h1 className="text-6xl font-bold text-center">About EcoPulse</h1>

        {/* High-level mission */}
        <p className="text-3xl text-brand-medium font-bold text-center">
          A service to engage the eco-conscious consumer by showing AI energy costs
          and promoting greener technology.
        </p>

        <p className="text-xl text-gray-700 leading-relaxed">
          Every time you scroll, request a chat response, or let a recommendation
          algorithm choose your next song, AI is running behind the scenes—drawing real electricity.
          In 2024, data centers used roughly <strong>1.5% of the world’s electricity</strong>, and
          that figure is projected to <strong>double by 2030</strong>.
        </p>

        {/* ─── Technological Improvement Image ─────────────────────────────────────────────── */}
        <section className="my-8">
          <h2 className="text-3xl font-semibold mb-4">Rapid Improvements in AI Technology</h2>
          <div className="flex justify-center">
            <img
              src="/Timeline-of-AI-generated-faces-updated.png"
              alt="Timeline of AI Generated Faces Over Time"
              className="max-w-full h-auto rounded shadow-md"
            />
          </div>
          <p className="mt-6 text-xl text-gray-700 leading-relaxed">
            AI image generators have gone from producing simple, pixelated faces in 2014 to creating  
            photorealistic images within a few years. In the same period, AI language and image recognition 
            skills have improved drastically, now beating humans in reading comprehension and image recognition tests.
          </p>
        </section>

        {/* ─── Homes-Powered Visualization ─────────────────────────────────────── */}
        <section className="my-8">
          <h2 className="text-3xl font-semibold mb-4">Putting It into Perspective</h2>
          <p className="text-xl text-gray-700 leading-relaxed mb-6">
            A tiny digit-recognition model uses almost no power, but a massive model like GPT-4
            requires enough electricity to power thousands of homes for an entire year:
          </p>
          <ul className="list-disc list-inside text-xl text-gray-700 space-y-4 mb-8">
            <li>
              <strong>MNIST Classifier (small model):</strong>{" "}
              Trains on about 10<sup>8</sup> FLOPs per pass. At ≈1 joule per 10<sup>9</sup> FLOPs,
              that’s ≈0.1 J (0.000000028 kWh) per step. Over 10 epochs (~600 steps each), total is under 
              0.002 kWh—about the energy to run a small LED bulb for a few minutes.
            </li>
            <li>
              <strong>GPT-4 (large model):</strong>{" "}
              Requires ≈2.15×10<sup>25</sup> FLOPs to train. At 0.6 J per 10<sup>9</sup> FLOPs, that’s 
              ≈6×10<sup>13</sup> J (≈57 million kWh). That is enough electricity to power roughly{" "}
              <strong>{totalHouses.toLocaleString()}</strong> U.S. homes for a full year.
            </li>
          </ul>

          <h3 className="text-2xl font-semibold mb-4">
            Visualizing {totalHouses.toLocaleString()} Homes
          </h3>
          <p className="text-xl text-gray-700 leading-relaxed mb-6">
            Each house icon below represents one U.S. home’s annual electricity usage. Together,
            these {totalHouses.toLocaleString()} icons convey the scale of GPT-4’s training energy in just 100 days.
          </p>
          {/* Increase grid gap to 2 for breathing room, reduce overflow height */}
          <div className="grid grid-cols-14 gap-2 max-h-64 overflow-y-auto">
            {housesArray.map((idx) => (
              <span key={idx} className="text-2xl leading-none">
                🏠
              </span>
            ))}
          </div>
        </section>

        {/* ─── How EcoPulse Helps ─────────────────────────────────────────────── */}
        <section className="space-y-6">
          <h2 className="text-3xl font-semibold">How EcoPulse Helps</h2>
          <p className="text-xl text-gray-700 leading-relaxed">
            On this site, you’ll find two main tools:
          </p>
          <ul className="list-disc list-inside text-xl text-gray-700 space-y-4">
            <li>
              <strong>Metrics Dashboard</strong>: View loss/accuracy charts over epochs plus
              “static” cards showing model size, total training energy (kWh), and training time.
              All numbers come directly from TensorBoard logs.
            </li>
            <li>
              <strong>Random Inference Demo</strong>: Run a pretrained digit-recognition model on  
              a random MNIST image and instantly see its predicted digit, how long it took,  
              and the tiny amount of energy it consumed. It’s a quick way to grasp why even small  
              models have energy costs.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
import { useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "./api/health";
import { About } from "./components/About";
import { AnalyzePanel } from "./components/AnalyzePanel";
import { Header } from "./components/Header";
import { HowItWorks } from "./components/HowItWorks";
import { Performance } from "./components/Performance";
import { StatusBanner } from "./components/StatusBanner";

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then((payload) => {
        if (!cancelled) {
          setHealth(payload);
          setError(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError("Could not reach the FreshNet API. Start the backend, then refresh.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const modelReady = health?.model.available === true;
  const modelMessage = error ?? health?.model.message ?? "The model is not available.";

  return (
    <div className="min-h-svh bg-[#f6f5f2] text-stone-900">
      <Header />
      <main className="mx-auto flex max-w-5xl flex-col gap-12 px-5 py-10">
        <section id="home" className="scroll-mt-24">
          <p className="text-sm uppercase tracking-[0.18em] text-[#1f4d3a]">Fresh or Spoiled</p>
          <h1 className="mt-3 max-w-3xl font-serif text-4xl leading-tight sm:text-5xl">
            Upload a fruit or vegetable image to classify its visible condition as Fresh or Spoiled.
          </h1>
          <p className="mt-4 max-w-2xl text-stone-700">
            The result is a visual estimate from the selected fused model. It does not measure safety,
            shelf life, or foods the model was not trained to describe.
          </p>
          <a
            href="#analyze"
            className="mt-6 inline-flex rounded-full bg-[#1f4d3a] px-5 py-3 text-sm font-medium text-white"
          >
            Analyze an image
          </a>
        </section>
        <StatusBanner health={health} error={error} loading={loading} />
        <AnalyzePanel modelReady={modelReady} modelMessage={modelMessage} />
        <HowItWorks />
        <Performance />
        <About />
      </main>
    </div>
  );
}

export default App;

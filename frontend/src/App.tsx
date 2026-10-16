import { useEffect, useRef, useState } from "react";
import { fetchHealth, type HealthResponse } from "./api/health";
import { About } from "./components/About";
import { AnalyzePanel } from "./components/AnalyzePanel";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { HowItWorks } from "./components/HowItWorks";
import { Performance } from "./components/Performance";
import { PhotoBand } from "./components/PhotoBand";
import { Produce } from "./components/Produce";
import { ProduceBelt } from "./components/ProduceBelt";
import { StatusBanner } from "./components/StatusBanner";

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const heroRef = useRef<HTMLElement>(null);

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

  // Reveal each section's parts once, as they enter the viewport.
  useEffect(() => {
    const targets = document.querySelectorAll<HTMLElement>("[data-reveal], .shutter");
    if (!("IntersectionObserver" in window)) {
      targets.forEach((element) => element.classList.add("is-in"));
      return;
    }
    // A fully clipped picture reports no intersection, so pictures are
    // watched through their parent box and revealed when it comes into view.
    const revealFor = new Map<Element, HTMLElement[]>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            revealFor.get(entry.target)?.forEach((element) => element.classList.add("is-in"));
            observer.unobserve(entry.target);
          }
        }
      },
      { rootMargin: "0px 0px -4% 0px", threshold: 0.02 },
    );
    targets.forEach((element) => {
      const watched = element.classList.contains("shutter") ? (element.parentElement ?? element) : element;
      revealFor.set(watched, [...(revealFor.get(watched) ?? []), element]);
      observer.observe(watched);
    });
    return () => observer.disconnect();
  }, []);

  // The hero produce leans slightly toward the pointer.
  useEffect(() => {
    const hero = heroRef.current;
    if (!hero || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }
    let frame = 0;
    function onMove(event: PointerEvent) {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const rect = hero!.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width - 0.5;
        const y = (event.clientY - rect.top) / rect.height - 0.5;
        hero!.style.setProperty("--mx", x.toFixed(3));
        hero!.style.setProperty("--my", y.toFixed(3));
      });
    }
    hero.addEventListener("pointermove", onMove);
    return () => {
      hero.removeEventListener("pointermove", onMove);
      cancelAnimationFrame(frame);
    };
  }, []);

  const modelReady = health?.model.available === true;
  const modelMessage = error ?? health?.model.message ?? "The model is not available.";

  return (
    <div className="min-h-svh">
      <Header />
      <main>
        <section
          id="home"
          ref={heroRef}
          className="relative scroll-mt-16 overflow-hidden"
          style={{ ["--mx" as string]: 0, ["--my" as string]: 0 }}
        >
          <div className="mx-auto grid max-w-[1440px] items-start gap-12 px-[var(--margin)] pb-[clamp(72px,9vw,140px)] pt-[clamp(32px,6vw,88px)] lg:grid-cols-[5fr_6fr]">
            <div className="relative z-10 pt-2">
              <h1 className="lines text-[length:var(--step-head)]">
                <span className="line">
                  <span data-line style={{ ["--i" as string]: 0 }}>
                    A second look
                  </span>
                </span>
                <span className="line">
                  <span data-line style={{ ["--i" as string]: 1 }}>
                    at the fruit bowl.
                  </span>
                </span>
              </h1>
              <p
                className="mt-6 max-w-[34ch] text-[length:var(--step-lead)] leading-snug text-[var(--ink)]"
                data-rise
                style={{ ["--i" as string]: 0 }}
              >
                Upload a fruit or vegetable image and FreshNet will say whether it looks Fresh or
                Spoiled.
              </p>
              <p
                className="mt-4 max-w-[46ch] text-[var(--ink)]"
                data-rise
                style={{ ["--i" as string]: 1 }}
              >
                It only talks about what the picture shows. It does not taste the food, date it, or tell
                you it is safe to eat.
              </p>
              <div className="mt-8" data-rise style={{ ["--i" as string]: 2 }}>
                <StatusBanner health={health} error={error} loading={loading} />
              </div>
            </div>

            <div id="analyze" className="relative scroll-mt-20">
              <div className="relative bg-[var(--field)] p-[clamp(20px,4vw,48px)]" data-rise style={{ ["--i" as string]: 1 }}>
                <AnalyzePanel modelReady={modelReady} modelMessage={modelMessage} checking={loading} />
              </div>

              <div
                className="pointer-events-none absolute -top-10 right-[-8px] z-20 hidden sm:block"
                style={{ transform: "translate(calc(var(--mx) * -22px), calc(var(--my) * -22px))" }}
              >
                <div className="drop" style={{ ["--i" as string]: 0 }}>
                  <div className="bob" style={{ ["--i" as string]: 0 }}>
                    <Produce kind="apple" size={118} />
                  </div>
                </div>
              </div>
              <div
                className="pointer-events-none absolute -bottom-9 -left-8 z-20 hidden sm:block"
                style={{ transform: "translate(calc(var(--mx) * 26px), calc(var(--my) * 18px))" }}
              >
                <div className="drop" style={{ ["--i" as string]: 1 }}>
                  <div className="bob" style={{ ["--i" as string]: 2 }}>
                    <Produce kind="lemon" size={104} />
                  </div>
                </div>
              </div>
              <div
                className="pointer-events-none absolute -right-6 bottom-24 z-20 hidden xl:block"
                style={{ transform: "translate(calc(var(--mx) * -30px), calc(var(--my) * 20px))" }}
              >
                <div className="drop" style={{ ["--i" as string]: 2 }}>
                  <div className="bob" style={{ ["--i" as string]: 4 }}>
                    <Produce kind="carrot" size={110} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div
            className="pointer-events-none absolute bottom-6 left-[var(--margin)] z-0 hidden lg:block"
            style={{ transform: "translate(calc(var(--mx) * 18px), calc(var(--my) * 12px))" }}
          >
            <div className="drop" style={{ ["--i" as string]: 3 }}>
              <div className="bob" style={{ ["--i" as string]: 5 }}>
                <Produce kind="broccoli" size={96} />
              </div>
            </div>
          </div>
        </section>

        <ProduceBelt />
        <HowItWorks />
        <Performance />
        <PhotoBand />
        <About />
      </main>
      <Footer />
    </div>
  );
}

export default App;

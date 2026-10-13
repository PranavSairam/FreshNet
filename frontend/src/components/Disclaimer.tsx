import { Info } from "lucide-react";

export function Disclaimer() {
  return (
    <aside className="flex gap-3 rounded-2xl border border-stone-300 bg-[#f8f4ea] px-4 py-3 text-sm text-stone-800">
      <Info size={18} className="mt-0.5 shrink-0 text-[#1f4d3a]" aria-hidden="true" />
      <p>
        FreshNet produces a <strong>visual freshness estimate</strong> from an image. It
        cannot tell you whether food is safe to eat, remaining shelf life, or a numeric
        freshness score derived from class probability.
      </p>
    </aside>
  );
}

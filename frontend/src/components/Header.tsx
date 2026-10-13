import { Leaf } from "lucide-react";

const links = [
  ["Home", "#home"],
  ["Analyze", "#analyze"],
  ["How it works", "#how-it-works"],
  ["Performance", "#performance"],
  ["About", "#about"],
] as const;

export function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-stone-200 bg-[#f6f5f2]/95 backdrop-blur">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-5 py-4">
        <a href="#home" className="flex items-center gap-3 rounded-lg">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#1f4d3a] text-white">
            <Leaf size={20} aria-hidden="true" />
          </span>
          <span>
            <span className="block text-xs font-medium uppercase tracking-[0.16em] text-[#1f4d3a]">
              Visual freshness
            </span>
            <span className="font-serif text-2xl text-stone-900">FreshNet</span>
          </span>
        </a>
        <nav aria-label="Page sections" className="flex flex-wrap gap-1 text-sm">
          {links.map(([label, href]) => (
            <a
              key={href}
              href={href}
              className="rounded-full px-3 py-2 text-stone-700 hover:bg-white hover:text-[#1f4d3a]"
            >
              {label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}

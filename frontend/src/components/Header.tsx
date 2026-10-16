const links = [
  ["Home", "#home"],
  ["Analyze", "#analyze"],
  ["How It Works", "#how-it-works"],
  ["Model Performance", "#performance"],
  ["About / Team", "#about"],
] as const;

export function Header() {
  return (
    <header className="sticky top-0 z-30 bg-[var(--bg)]">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-x-6 gap-y-1 px-[var(--margin)] py-3">
        <a href="#home" className="display text-[length:var(--step-lead)] leading-none text-[var(--ink)]">
          FreshNet
        </a>
        <nav aria-label="Page sections" className="order-3 -mx-2 flex w-full flex-wrap gap-x-1 md:order-none md:mx-0 md:w-auto">
          {links.map(([label, href]) => (
            <a
              key={href}
              href={href}
              className="link shrink-0 px-2 py-2 text-[length:var(--step-small)] text-[var(--ink)]"
            >
              {label}
            </a>
          ))}
        </nav>
        <a href="#analyze" className="button hidden sm:inline-flex">
          Try a photo
        </a>
      </div>
    </header>
  );
}

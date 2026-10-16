const credits = [
  {
    title: "Radish catalogue page",
    by: "John Lewis Childs (Firm), Henry G. Gilbert Nursery and Seed Trade Catalog Collection, 1901. No known copyright restrictions.",
    href: "https://commons.wikimedia.org/wiki/File:Childs%27_rare_flowers,_vegetables_and_fruits_(1901)_(20580503056).jpg",
  },
  {
    title: "Vegetable stall",
    by: "Acabashi, CC BY-SA 4.0, via Wikimedia Commons.",
    href: "https://commons.wikimedia.org/wiki/File:Outdoor_market_fruit_and_vegetable_stall_Market_Place_Romford_London_01.jpg",
  },
  {
    title: "Fruit stall",
    by: "Acabashi, CC BY-SA 4.0, via Wikimedia Commons.",
    href: "https://commons.wikimedia.org/wiki/File:Outdoor_market_fruit_stall_Market_Place_Romford_London_01.jpg",
  },
  {
    title: "Roxbury Russet",
    by: "Ed Yourdon, CC BY-SA 2.0, via Wikimedia Commons.",
    href: "https://commons.wikimedia.org/w/index.php?curid=11873053",
  },
];

export function Footer() {
  return (
    <footer className="overflow-hidden bg-[var(--field)] pt-[clamp(56px,8vw,120px)] text-[var(--field-ink)] on-field">
      <div className="mx-auto grid max-w-[1440px] gap-10 px-[var(--margin)] md:grid-cols-[1fr_1.4fr]">
        <p className="max-w-sm text-[length:var(--step-lead)] leading-snug" data-reveal>
          Visual estimates only. FreshNet does not date food, taste it, or certify that it is safe.
        </p>
        <div data-reveal style={{ ["--i" as string]: 1 }}>
          <p className="font-medium">Picture credits</p>
          <ul className="mt-3 list-none space-y-2 p-0 text-[length:var(--step-small)] text-white/80">
            {credits.map((credit) => (
              <li key={credit.title}>
                <a href={credit.href} className="link text-white" rel="noreferrer" target="_blank">
                  {credit.title}
                </a>
                . {credit.by}
              </li>
            ))}
          </ul>
          <p className="mt-4 text-[length:var(--step-small)] text-white/70">
            The fruit and vegetable drawings on this page were made for FreshNet.
          </p>
        </div>
      </div>
      <p
        className="display mt-[clamp(40px,6vw,96px)] select-none px-[var(--margin)] text-center text-[clamp(5rem,23vw,22rem)] leading-[0.8] text-[var(--field-soft)]"
        aria-hidden="true"
      >
        FreshNet
      </p>
    </footer>
  );
}

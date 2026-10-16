const team = [
  ["PRANAV SAIRAM", "RA2311026010195"],
  ["SANTHOSH G", "RA2311026010196"],
  ["V T S MUKUNDAN", "RA2311026010209"],
] as const;

function initials(name: string) {
  return name
    .split(" ")
    .filter((part) => part.length > 0)
    .map((part) => part[0])
    .slice(0, 2)
    .join("");
}

export function About() {
  return (
    <section id="about" aria-labelledby="about-title" className="scroll-mt-16 py-[clamp(80px,11vw,180px)]">
      <div className="mx-auto grid max-w-[1440px] items-start gap-12 px-[var(--margin)] lg:grid-cols-[5fr_6fr] lg:gap-[8%]">
        <div className="shutter drift relative aspect-[3/4] w-full overflow-hidden lg:sticky lg:top-24">
          <img
            src="/pictures/radish-plate-1901.jpg"
            alt="A pink-printed radish page from a 1901 seed catalogue, with drawings of long and round radishes"
            width={1400}
            height={2020}
            loading="lazy"
            className="absolute inset-0 h-full w-full object-cover"
            style={{ objectPosition: "top" }}
          />
        </div>

        <div className="max-w-[540px] lg:pt-[10vh]">
          <h2 id="about-title" className="mb-8 text-[length:var(--step-head)]" data-reveal>
            About / Team
          </h2>
          <div className="space-y-5 leading-[1.55] text-[var(--ink)]">
            <p data-reveal>We are three students, and FreshNet is our academic project.</p>
            <p data-reveal style={{ ["--i" as string]: 1 }}>
              The question was small: from one photo, does a fruit or vegetable look fresh or spoiled?
              We kept it to those two answers on purpose. A photo can show colour, bruising and mould.
              It cannot show how old the food is or whether it is safe to eat.
            </p>
            <p data-reveal style={{ ["--i" as string]: 2 }}>
              We trained three models, chose the best one on validation images, and scored the
              held-out test set once. Along the way we set aside 18,266 of 71,318 images we could not
              trust, and we say so here instead of hiding it.
            </p>
            <p data-reveal style={{ ["--i" as string]: 3 }}>
              Please treat every answer as a second look,{" "}
              <mark className="bg-[#c9d99f] px-1 text-[var(--ink)]">not a verdict</mark>. If something
              looks or smells wrong to you, believe yourself.
            </p>
          </div>

          <p className="mt-10 text-[length:var(--step-lead)] italic" data-reveal>
            The FreshNet team
          </p>
          <p className="text-[length:var(--step-small)] text-[var(--muted)]" data-reveal>
            Academic project
          </p>

          <ul className="mt-8 flex list-none flex-col gap-4 p-0">
            {team.map(([name, id], index) => (
              <li key={id} className="flex items-center gap-4" data-reveal style={{ ["--i" as string]: index }}>
                <span
                  aria-hidden="true"
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--accent)] text-[length:var(--step-small)] text-white"
                >
                  {initials(name)}
                </span>
                <span>
                  <span className="block font-medium">{name}</span>
                  <span className="block text-[length:var(--step-small)] text-[var(--muted)]">{id}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

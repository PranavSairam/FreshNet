const team = [
  ["PRANAV SAIRAM", "RA2311026010195"],
  ["SANTHOSH G", "RA2311026010196"],
  ["V T S MUKUNDAN", "RA2311026010209"],
];

export function About() {
  return (
    <section id="about" className="scroll-mt-24">
      <h2 className="font-serif text-3xl">About</h2>
      <p className="mt-2 max-w-2xl text-stone-700">
        FreshNet is a local academic project for visual food-freshness classification. Upload a fruit
        or vegetable image. The model estimates whether that image looks Fresh or Spoiled.
      </p>
      <ul className="mt-6 grid gap-3 sm:grid-cols-3">
        {team.map(([name, id]) => (
          <li key={id} className="rounded-2xl border border-stone-200 bg-white p-4">
            <p className="font-medium">{name}</p>
            <p className="mt-1 text-sm text-stone-600">{id}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}

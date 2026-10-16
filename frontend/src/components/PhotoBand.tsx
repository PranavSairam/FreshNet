export function PhotoBand() {
  return (
    <section aria-label="A fruit stall at a market" className="bg-[var(--bg)]">
      <div className="shutter drift relative aspect-[4/3] w-full overflow-hidden md:aspect-[21/9]">
        <img
          src="/pictures/fruit-stall.jpg"
          alt="A market stall of oranges, grapes, apples, bananas and watermelons in punnets and crates"
          width={1500}
          height={1125}
          loading="lazy"
          className="absolute inset-0 h-full w-full object-cover"
          style={{ objectPosition: "center 60%" }}
        />
      </div>
    </section>
  );
}

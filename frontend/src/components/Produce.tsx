import type { CSSProperties } from "react";

export type ProduceKind = "apple" | "tomato" | "lemon" | "carrot" | "broccoli" | "banana";
export type ProduceState = "fresh" | "spoiled";

type Palette = { fresh: string; rot: string; leaf?: string };

const palettes: Record<ProduceKind, Palette> = {
  apple: { fresh: "#c63a2d", rot: "#7a4a2c" },
  tomato: { fresh: "#d9432e", rot: "#7c4a35" },
  lemon: { fresh: "#f0cc3a", rot: "#8e7a31" },
  carrot: { fresh: "#ee8a2b", rot: "#8c5b38", leaf: "#4f9a3a" },
  broccoli: { fresh: "#3f8f3a", rot: "#8d8b3f", leaf: "#6aa84f" },
  banana: { fresh: "#f3d44f", rot: "#6e4c2b" },
};

export const produceNames: Record<ProduceKind, string> = {
  apple: "Apple",
  tomato: "Tomato",
  lemon: "Lemon",
  carrot: "Carrot",
  broccoli: "Broccoli",
  banana: "Banana",
};

type Props = {
  kind: ProduceKind;
  state?: ProduceState;
  size?: number;
  className?: string;
};

export function Produce({ kind, state = "fresh", size = 120, className }: Props) {
  const palette = palettes[kind];
  const style = {
    width: size,
    height: size,
    "--fresh": palette.fresh,
    "--rot": palette.rot,
    "--fresh-leaf": palette.leaf ?? "#5f9a3c",
  } as CSSProperties;

  return (
    <svg
      viewBox="0 0 120 120"
      className={`produce ${className ?? ""}`}
      data-state={state}
      style={style}
      aria-hidden="true"
      focusable="false"
    >
      {kind === "apple" && (
        <>
          <path
            data-fill="body"
            d="M60 32c-10-8-34-8-40 14-6 24 8 56 26 62 8 3 11-2 14-2s6 5 14 2c18-6 32-38 26-62-6-22-30-22-40-14z"
          />
          <path d="M60 32c0-9 3-17 9-23" stroke="#5a4328" strokeWidth="5" strokeLinecap="round" fill="none" />
          <path data-fill="leaf" d="M64 22c10-12 26-10 30-6-6 12-20 14-30 6z" />
          <path d="M33 54c1-9 7-15 14-17" stroke="#fff" strokeWidth="5" strokeLinecap="round" fill="none" opacity=".45" />
          <g data-fill="spots" fill="#3d2a1c">
            <circle cx="46" cy="70" r="6" />
            <circle cx="78" cy="58" r="5" />
            <circle cx="70" cy="86" r="7" />
            <circle cx="54" cy="92" r="3.5" />
          </g>
        </>
      )}

      {kind === "tomato" && (
        <>
          <path
            data-fill="body"
            d="M60 28C30 26 12 48 14 70c2 24 22 40 46 40s44-16 46-40c2-22-16-44-46-42z"
          />
          <path
            data-fill="leaf"
            d="M60 22l8 12 14-6-6 14 14 4-14 8-8-10-8 10-14-6 10-10-12-8 14 0-4-14z"
          />
          <path d="M30 60c2-10 9-17 17-19" stroke="#fff" strokeWidth="5" strokeLinecap="round" fill="none" opacity=".4" />
          <g data-fill="spots" fill="#3d2a1c">
            <circle cx="40" cy="84" r="6" />
            <circle cx="80" cy="76" r="7" />
            <circle cx="64" cy="96" r="4" />
          </g>
        </>
      )}

      {kind === "lemon" && (
        <>
          <path
            data-fill="body"
            d="M10 62c0-6 5-8 10-8 6-20 24-30 44-30s36 12 40 32c5 0 10 2 10 6s-5 6-10 6c-4 20-20 32-40 32s-38-12-44-32c-5 0-10-2-10-6z"
          />
          <path d="M38 44c6-8 14-12 24-12" stroke="#fff" strokeWidth="5" strokeLinecap="round" fill="none" opacity=".5" />
          <path data-fill="leaf" d="M62 28c6-12 18-16 26-12-2 12-14 18-26 12z" />
          <g data-fill="spots" fill="#47341f">
            <circle cx="50" cy="72" r="6" />
            <circle cx="78" cy="64" r="5" />
            <circle cx="66" cy="84" r="3.5" />
          </g>
        </>
      )}

      {kind === "carrot" && (
        <>
          <path data-fill="leaf" d="M76 30c-6-16 0-26 6-28 2 10 6 16 2 28z" />
          <path data-fill="leaf" d="M70 32c-14-10-26-8-32-2 8 8 20 12 32 2z" />
          <path data-fill="leaf" d="M82 34c14-8 24-4 30 4-10 6-22 6-30-4z" />
          <path data-fill="body" d="M62 34c10-6 24-2 26 10 2 14-4 26-14 40-10 14-40 30-56 30 0-16 18-46 30-62 4-8 8-14 14-18z" />
          <path d="M32 78l12-6M44 62l10-4M56 48l8-4" stroke="#fff" strokeWidth="3" strokeLinecap="round" opacity=".35" />
          <g data-fill="spots" fill="#3d2a1c">
            <circle cx="58" cy="60" r="5" />
            <circle cx="44" cy="82" r="6" />
            <circle cx="72" cy="46" r="3.5" />
          </g>
        </>
      )}

      {kind === "broccoli" && (
        <>
          <path d="M52 66h16l6 44H46z" fill="#9bbd6f" />
          <g data-fill="body">
            <circle cx="36" cy="52" r="22" />
            <circle cx="62" cy="38" r="26" />
            <circle cx="86" cy="54" r="22" />
            <circle cx="60" cy="62" r="22" />
          </g>
          <g data-fill="leaf" opacity=".55">
            <circle cx="30" cy="46" r="6" />
            <circle cx="56" cy="30" r="7" />
            <circle cx="82" cy="46" r="6" />
            <circle cx="62" cy="60" r="6" />
          </g>
          <g data-fill="spots" fill="#4a3a1e">
            <circle cx="38" cy="58" r="6" />
            <circle cx="72" cy="30" r="5" />
            <circle cx="88" cy="62" r="5" />
          </g>
        </>
      )}

      {kind === "banana" && (
        <>
          <path
            data-fill="body"
            d="M18 40c8 38 34 62 76 52 8-2 14-8 14-8-26 2-52-12-62-46-2-6-8-6-12-4-8 2-18 2-16 6z"
          />
          <path d="M18 40c-2-8 0-14 6-18 4 6 6 14 4 22z" fill="#6a553a" />
          <path d="M104 86c6-2 10-6 10-6l-6-6c-4 4-8 8-14 8z" fill="#4b3b27" />
          <path d="M30 52c10 22 28 34 48 36" stroke="#fff" strokeWidth="4" strokeLinecap="round" fill="none" opacity=".4" />
          <g data-fill="spots" fill="#2f2114">
            <circle cx="48" cy="66" r="4" />
            <circle cx="64" cy="82" r="5" />
            <circle cx="84" cy="88" r="3.5" />
            <circle cx="38" cy="50" r="3" />
          </g>
        </>
      )}
    </svg>
  );
}

import { Upload } from "lucide-react";

type Props = {
  enabled: boolean;
};

export function UploadPanel({ enabled }: Props) {
  return (
    <section className="rounded-3xl border border-stone-300 bg-white p-6 shadow-sm">
      <div className="mb-4 flex items-center gap-2 text-[#1f4d3a]">
        <Upload size={18} aria-hidden="true" />
        <h2 className="font-serif text-xl text-stone-900">Image upload</h2>
      </div>
      <label
        className={`flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-10 text-center ${
          enabled
            ? "border-[#1f4d3a] bg-emerald-50/40 text-stone-800"
            : "cursor-not-allowed border-stone-300 bg-stone-50 text-stone-500"
        }`}
      >
        <input type="file" accept="image/jpeg,image/png,image/webp" disabled={!enabled} className="hidden" />
        <p className="font-medium">
          {enabled ? "Choose a food image" : "Upload disabled until a trained model is installed"}
        </p>
        <p className="mt-2 max-w-md text-sm">
          Supported files: JPEG, PNG, WebP. The predictor is not connected in this stage.
        </p>
      </label>
    </section>
  );
}

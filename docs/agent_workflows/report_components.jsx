/**
 * VPA report — React + Tailwind reference implementation of the production-grade
 * scientific-platform redesign (Linear / Stripe / Datadog / Benchling register).
 *
 * This mirrors the shipped vanilla-CSS report in `reporting/render.py`. It is the
 * canonical design spec: restrained, neutral-default, traceable. Drop into any
 * React + Tailwind app. No plugins required.
 *
 * DESIGN SYSTEM (the rules these components encode)
 * ─────────────────────────────────────────────────
 * Colour            Neutral surfaces by default. Accent (indigo-600) ONLY for
 *                   the selected row + active nav. Semantic status dots
 *                   (emerald=ok / amber=watch / rose=critical) are the only
 *                   other colour. No decorative colour, no success-green
 *                   emphasis, no gradients.
 * Typography        Hierarchy via size/weight, not colour. Values lead (17–19px
 *                   semibold, tabular-nums); labels are 11px muted. Monospace
 *                   for identifiers + sequences.
 * Spacing           8px base. Whitespace + hairline dividers group content —
 *                   not borders. Avoid nested cards (max one level: leaf tiles).
 * Borders           1px slate-200 hairlines; dividers over boxes.
 * Elevation         Effectively none. A faint shadow only on a true overlay.
 * Badges            Outline, neutral. Reserved; not decorative.
 * Provenance        Every confidence metric carries "· Source" inline muted
 *                   text (the producing model), never a coloured pill.
 * Selected state    Left accent bar + faint accent tint (interaction).
 * Final state       ✓ rank glyph + neutral "FINAL" outline label + ink weight
 *                   (semantic). DISTINCT from selected — they compose.
 */
import React, { useState } from "react";

/* ── tokens ──────────────────────────────────────────────────────────────── */
const DOT = {
  ok: "bg-emerald-500", good: "bg-emerald-500", completed: "bg-emerald-500",
  watch: "bg-amber-500", queued: "bg-amber-500", running: "bg-amber-500",
  critical: "bg-rose-500", failed: "bg-rose-500", poor: "bg-rose-500",
  neutral: "bg-slate-300",
};
const Dot = ({ tone = "neutral", className = "" }) => (
  <span className={`inline-block h-[7px] w-[7px] flex-none rounded-full ${DOT[tone] ?? DOT.neutral} ${className}`} />
);

/* ── Provenance — "· Source" inline, never a coloured badge ─────────────────── */
export const Source = ({ tool }) =>
  tool ? <span className="font-normal text-slate-400">· {tool}</span> : null;

/* ── Metric row (Scores / QC / details) ─────────────────────────────────────
 * @param {{label:string, value:string, unit?:string, source?:string, tone?:string}} */
export function MetricRow({ label, value, unit, source, tone }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-t border-slate-100 py-1.5 first:border-t-0">
      <span className="flex items-baseline gap-1.5 text-[13px] text-slate-700">
        {tone && <Dot tone={tone} className="self-center" />}
        {label} <Source tool={source} />
      </span>
      <span className="text-[13px] font-semibold tabular-nums text-slate-900">
        {value}{unit ? <span className="ml-0.5 text-slate-400">{unit}</span> : null}
      </span>
    </div>
  );
}

/* ── Executive summary — stat strip + two-column signal band ─────────────────── */
export function ExecutiveSummary({ headline, stats, confidence, pipeline }) {
  return (
    <section>
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Executive summary</p>
      <h2 className="mt-1 text-[19px] font-semibold text-slate-900">{headline}</h2>

      <div className="mt-4 flex flex-wrap border-y border-slate-200 py-3.5">
        {stats.map((s, i) => (
          <div key={s.label} className={`px-5 ${i === 0 ? "pl-0" : "border-l border-slate-200"}`}>
            <div className="flex items-center gap-1.5 text-[19px] font-semibold tabular-nums text-slate-900">
              {s.tone && <Dot tone={s.tone} />}{s.value}
            </div>
            <div className="mt-0.5 text-[11px] text-slate-500">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-8 md:grid-cols-2">
        <div>
          <h3 className="mb-2.5 border-b border-slate-200 pb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Confidence <span className="font-medium normal-case tracking-normal text-slate-400">· top candidate</span>
          </h3>
          <div className="flex flex-col gap-2.5">
            {confidence.map((m) => (
              <div key={m.label} className="flex items-baseline gap-2.5">
                <Dot tone={m.tone} className="self-center" />
                <span className="min-w-[46px] text-[17px] font-semibold tabular-nums text-slate-900">{m.value}</span>
                <span className="text-[13px] text-slate-700">{m.label} <Source tool={m.source} /></span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h3 className="mb-2.5 border-b border-slate-200 pb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Pipeline <span className="font-medium normal-case tracking-normal text-slate-400">· {pipeline.done}/{pipeline.nodes.length} completed</span>
          </h3>
          <div className="flex flex-col gap-2">
            {pipeline.nodes.map((n) => (
              <div key={n.name} className="flex items-center gap-2 font-mono text-[12.5px] text-slate-700">
                <Dot tone={n.tone} /> {n.name}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── Candidate result browser — dense table/list hybrid ──────────────────────
 * Final = ✓ + neutral FINAL label (semantic). Selected = left bar + tint
 * (interaction). The two compose and never share a treatment.
 * @param {{ groups: {title:string, count:number, collapsed?:boolean,
 *            rows:{id:string, rank:number|null, final?:boolean, status?:string,
 *                  value?:string, metric?:string, source?:string}[]}[],
 *           value?:string, onSelect?:(id)=>void }} */
export function CandidateBrowser({ groups, defaultValue, onSelect }) {
  const first = defaultValue ?? groups.flatMap((g) => g.rows)[0]?.id ?? null;
  const [sel, setSel] = useState(first);
  const pick = (id) => { setSel(id); onSelect?.(id); };

  const Row = ({ r }) => {
    const selected = r.id === sel;
    return (
      <button
        type="button" role="tab" aria-selected={selected} onClick={() => pick(r.id)}
        className={`grid w-full grid-cols-[34px_minmax(0,1fr)_auto] items-center gap-2.5 border-b border-l-2 border-slate-100 px-3 py-2 text-left transition-colors
          ${selected ? "border-l-indigo-600 bg-indigo-600/5" : "border-l-transparent hover:bg-slate-50"}`}
      >
        <span className="text-center">
          {r.final
            ? <span className="text-[14px] font-bold text-slate-900" title="Final / submitted">✓</span>
            : <span className="text-[13px] font-semibold tabular-nums text-slate-400">{r.rank ?? "·"}</span>}
        </span>
        <span className={`flex min-w-0 items-center gap-2 font-mono text-[12.5px] text-slate-900 ${selected ? "font-semibold" : "font-medium"}`}>
          <span className="truncate">{r.id}</span>
          {r.final && (
            <span className="flex-none rounded border border-slate-300 bg-slate-50 px-1.5 py-px text-[9px] font-bold uppercase tracking-wide text-slate-700">
              Final
            </span>
          )}
        </span>
        <span className="flex flex-col items-end">
          {r.value
            ? <><span className="text-[13px] font-semibold tabular-nums text-slate-900">{r.value}</span>
                <span className="whitespace-nowrap text-[10px] text-slate-400">{r.metric}{r.source ? ` · ${r.source}` : ""}</span></>
            : <span className="text-[13px] text-slate-300">—</span>}
        </span>
      </button>
    );
  };

  return (
    <div>
      <div className="grid grid-cols-[34px_minmax(0,1fr)_auto] items-center gap-2.5 border-b border-slate-200 px-3 pb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
        <span className="text-center">Rank</span><span>Candidate</span><span className="text-right">Score</span>
      </div>
      {groups.map((g) => (
        <section key={g.title} className="mt-3.5 first:mt-2.5">
          <div className="flex items-center gap-2 px-3 py-1 text-[10.5px] font-bold uppercase tracking-wider text-slate-500">
            {g.title} <span className="font-medium text-slate-400">· {g.count}</span>
          </div>
          {g.rows.map((r) => <Row key={r.id} r={r} />)}
        </section>
      ))}
    </div>
  );
}

/* ── Candidate detail header — authoritative final, not celebratory ─────────── */
export function CandidateDetailHeader({ id, status = "ranked", note }) {
  const chip = status === "final"
    ? "bg-slate-900 text-white border-slate-900"
    : status === "failed"
    ? "bg-rose-50 text-rose-700 border-rose-200"
    : "bg-slate-50 text-slate-600 border-slate-300";
  return (
    <div className="border-b border-slate-200 pb-4">
      <span className={`inline-block rounded border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${chip}`}>
        {status}
      </span>
      <h3 className="mt-1.5 font-mono text-[19px] font-semibold text-slate-900">{id}</h3>
      {note && (
        <p className="mt-1.5 border-l-2 border-slate-300 pl-2.5 text-[12px] text-slate-500">{note}</p>
      )}
    </div>
  );
}

/* ── Example ─────────────────────────────────────────────────────────────────
const groups = [
  { title: "Final / submitted", count: 1, rows: [
    { id: "cand_b7d60b8ca687", rank: 5, final: true, value: "62.0", metric: "pLDDT", source: "ESM-3" }]},
  { title: "Ranked alternatives", count: 4, rows: [
    { id: "cand_055", rank: 1, value: "91.0", metric: "pLDDT", source: "ESM-3" },
    { id: "cand_19b", rank: 2, value: "85.0", metric: "pLDDT", source: "ESM-3" }]},
];
<ExecutiveSummary
  headline="Completed in 5.8 min · 3 candidates · top pLDDT 86, ipTM 0.05 · no warnings"
  stats={[{label:"Status",value:"Completed",tone:"completed"},{label:"Runtime",value:"5.8 min"},
          {label:"Candidates",value:"3"},{label:"Pipeline nodes",value:"4"}]}
  confidence={[{label:"pLDDT",value:"86",source:"BindCraft",tone:"ok"},
               {label:"ipTM",value:"0.05",source:"BindCraft",tone:"critical"}]}
  pipeline={{done:4, nodes:[{name:"bindcraft",tone:"completed"},{name:"rank_designs",tone:"completed"}]}}
/>
<CandidateBrowser groups={groups} onSelect={(id) => filterTo(id)} />
──────────────────────────────────────────────────────────────────────────── */

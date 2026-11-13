/**
 * TADA designer web search for pi.
 *
 * Calls the configured unified search pool (POST,
 * Exa primary, Tavily fallback). This is the only tool the task designer gets:
 * load with `pi --no-extensions --no-builtin-tools -e <this file> --tools web_search`.
 * No local file, shell or memory access is exposed here.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { configuredValue } from "./local_config.ts";
const TIMEOUT_MS = 90_000;
const MAX_CONTENT_CHARS = 2_000;

type Hit = { title?: string; url?: string; content?: string; text?: string; published_date?: string };

export async function search(query: string, limit = 5, engine = "auto", signal?: AbortSignal) {
  const timeout = AbortSignal.timeout(TIMEOUT_MS);
  const res = await fetch(configuredValue("search_url"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, limit, engine, include_contents: true }),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  const body = await res.text();
  if (!res.ok) throw new Error(`search HTTP ${res.status}: ${body.slice(0, 300)}`);
  const data = JSON.parse(body) as { provider?: string; results?: Hit[] };
  const hits = data.results ?? [];
  const text = hits.length
    ? hits
        .map((h, i) => {
          const content = (h.content ?? h.text ?? "").slice(0, MAX_CONTENT_CHARS);
          const date = h.published_date ? ` (${h.published_date})` : "";
          return `[${i + 1}] ${h.title ?? "(untitled)"}${date}\n${h.url ?? ""}\n${content}`;
        })
        .join("\n\n")
    : "No results.";
  return { text, provider: data.provider ?? null, count: hits.length };
}

export default function (pi: ExtensionAPI) {
  // Enforce: the designer session exposes exactly this one tool. pi logs (but does not abort on)
  // extension errors, so restrict the active set instead of throwing, then re-check it.
  pi.on?.("session_start", () => {
    const names = () => (pi.getActiveTools?.() ?? []).map((t: any) => (typeof t === "string" ? t : t.name)).sort();
    const before = names();
    if (before.length !== 1 || before[0] !== "web_search") pi.setActiveTools?.(["web_search"]);
    const after = names();
    if (process.env.TADA_TOOL_AUDIT) console.error(`TADA_ACTIVE_TOOLS before=${JSON.stringify(before)} after=${JSON.stringify(after)}`);
    if (after.length !== 1 || after[0] !== "web_search") {
      throw new Error(`TADA designer must expose only web_search; active tools: ${after.join(",")}`);
    }
  });

  pi.registerTool({
    name: "web_search",
    label: "Web Search",
    description:
      "Search the public web (Exa, Tavily fallback) and return titles, URLs and page excerpts. " +
      "Use for occupational practice, regulatory or professional procedures, document formats, " +
      "and official dataset or codebook documentation. Never search for per-record labels or answers.",
    promptSnippet: "Search the public web for professional practice and official dataset documentation",
    parameters: Type.Object({
      query: Type.String({ description: "Search query." }),
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 10, description: "Results to return (default 5)." })),
      engine: Type.Optional(
        Type.Union([Type.Literal("auto"), Type.Literal("exa"), Type.Literal("tavily")], {
          description: "Search engine (default auto).",
        }),
      ),
    }),
    async execute(_id, params, signal) {
      const r = await search(params.query, params.limit ?? 5, params.engine ?? "auto", signal);
      return {
        content: [{ type: "text" as const, text: r.text }],
        details: { provider: r.provider, count: r.count, query: params.query },
      };
    },
  });
}

/**
 * TADA designer tools for pi: web search plus confined task-file writing.
 *
 *  web_search       -> local unified search pool (re-exported from tada_web_search.ts)
 *  write_task_file  -> write one text file inside $TADA_TASK_ROOT (no traversal, no symlinks, size cap)
 *  read_task_file   -> read back a file the designer wrote inside $TADA_TASK_ROOT
 *  list_task_files  -> list files under $TADA_TASK_ROOT
 *
 * Load with: pi --no-extensions --no-builtin-tools -e <this file>
 *            --tools web_search,write_task_file,read_task_file,list_task_files
 * On session_start the active set is forced to exactly these four tools (pi built-ins removed).
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import * as fs from "node:fs";
import * as path from "node:path";
import { search } from "./tada_web_search.ts";
import { spawnSync } from "node:child_process";
import { configuredValue } from "./local_config.ts";

const HERE = path.dirname(new URL(import.meta.url).pathname);
const checkPython = () => process.env.TADA_CHECK_PYTHON || configuredValue("datasets_python");

const TASK_ROOT = process.env.TADA_TASK_ROOT ? path.resolve(process.env.TADA_TASK_ROOT) : "";
const MAX_FILE_BYTES = 400_000;
const V5 = process.env.TADA_DESIGNER_V5 === "1";
const V5_FILES = new Set(["dependency_graph.json", "scenario_spec.json", "instruction.md", "design.md"]);
export const ALLOWED = V5 ? ["check_task_files", "list_task_files", "read_task_file", "write_task_file"] :
  ["list_task_files", "read_task_file", "web_search", "write_task_file"];

/** Resolve a designer-supplied relative path inside root; throws on escape or symlink. */
export function safePath(rel: string, root = TASK_ROOT): string {
  if (!root) throw new Error("TADA_TASK_ROOT is not set");
  if (typeof rel !== "string" || !rel.trim() || path.isAbsolute(rel) || rel.includes("\0")) throw new Error(`invalid path: ${rel}`);
  const norm = path.normalize(rel);
  if (norm.split(path.sep).includes("..")) throw new Error(`path escapes task root: ${rel}`);
  const target = path.resolve(root, norm);
  if (!target.startsWith(root + path.sep)) throw new Error(`path escapes task root: ${rel}`);
  let cur = root;
  for (const part of path.relative(root, target).split(path.sep)) {
    cur = path.join(cur, part);
    if (fs.existsSync(cur) && fs.lstatSync(cur).isSymbolicLink()) throw new Error(`symlink in path: ${rel}`);
  }
  return target;
}

function walk(dir: string, root: string, out: string[]) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, root, out);
    else if (e.isFile()) out.push(`${path.relative(root, p)}  (${fs.statSync(p).size} bytes)`);
  }
}

export default function (pi: ExtensionAPI) {
  if (!TASK_ROOT) throw new Error("TADA_TASK_ROOT must be set for the designer tools");

  pi.on?.("session_start", () => {
    const names = () => (pi.getActiveTools?.() ?? []).map((t: any) => (typeof t === "string" ? t : t.name)).sort();
    const before = names();
    if (JSON.stringify(before) !== JSON.stringify(ALLOWED)) pi.setActiveTools?.(ALLOWED);
    const after = names();
    if (process.env.TADA_TOOL_AUDIT) console.error(`TADA_ACTIVE_TOOLS before=${JSON.stringify(before)} after=${JSON.stringify(after)}`);
    if (JSON.stringify(after) !== JSON.stringify(ALLOWED)) throw new Error(`TADA designer tool set mismatch: ${after.join(",")}`);
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
        Type.Union([Type.Literal("auto"), Type.Literal("exa"), Type.Literal("tavily")], { description: "Search engine (default auto)." }),
      ),
    }),
    async execute(_id, params, signal) {
      const r = await search(params.query, params.limit ?? 5, params.engine ?? "auto", signal);
      return { content: [{ type: "text" as const, text: r.text }], details: { provider: r.provider, count: r.count, query: params.query } };
    },
  });

  pi.registerTool({
    name: "write_task_file",
    label: "Write Task File",
    description:
      "Write (create or overwrite) one text file of the Harbor task you are designing. " +
      "path is relative to the task root, e.g. 'build/build.py' or 'design.md'. Writes outside the task root are refused.",
    promptSnippet: "Write one file of the task package (relative path)",
    parameters: Type.Object({
      path: Type.String({ description: "Relative path inside the task root." }),
      content: Type.String({ description: "Full file contents." }),
    }),
    async execute(_id, params) {
      if (V5 && !V5_FILES.has(params.path)) throw new Error("v5 permits only the four declared artifacts");
      const target = safePath(params.path);
      const bytes = Buffer.byteLength(params.content, "utf8");
      if (bytes > MAX_FILE_BYTES) throw new Error(`file too large (${bytes} bytes > ${MAX_FILE_BYTES})`);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, params.content, { encoding: "utf8" });
      return { content: [{ type: "text" as const, text: `wrote ${params.path} (${bytes} bytes)` }], details: { path: params.path, bytes } };
    },
  });

  pi.registerTool({
    name: "read_task_file",
    label: "Read Task File",
    description: "Read back a file you wrote inside the task root.",
    promptSnippet: "Read back one of your task files",
    parameters: Type.Object({ path: Type.String({ description: "Relative path inside the task root." }) }),
    async execute(_id, params) {
      if (V5 && !V5_FILES.has(params.path)) throw new Error("v5 permits only the four declared artifacts");
      const target = safePath(params.path);
      if (!fs.existsSync(target)) throw new Error(`not found: ${params.path}`);
      return { content: [{ type: "text" as const, text: fs.readFileSync(target, "utf8") }], details: { path: params.path } };
    },
  });

  if (V5) pi.registerTool({
    name: "check_task_files",
    label: "Check Task Files",
    description:
      "Run the compiler's format check on your four files (schema, templates, carriers, copies). " +
      "Returns ok or the first structural error. It never runs the task or reveals answers. Call it after writing and fix every error before finishing.",
    promptSnippet: "Check the four files against the compiler contract",
    parameters: Type.Object({}),
    async execute() {
      const r = spawnSync(checkPython(), ["-m", "tada_taskgen.designer_check", TASK_ROOT], {
        cwd: HERE, encoding: "utf8", timeout: 120_000,
        env: { PATH: process.env.PATH ?? "", HOME: process.env.HOME ?? "", PYTHONPATH: path.join(HERE, "v5"), PYTHONDONTWRITEBYTECODE: "1" },
      });
      let text = (r.stdout || "").trim().split("\n").pop() || "";
      try { JSON.parse(text); } catch { text = JSON.stringify({ ok: false, error: "format check could not run" }); }
      return { content: [{ type: "text" as const, text }], details: {} };
    },
  });

  pi.registerTool({
    name: "list_task_files",
    label: "List Task Files",
    description: "List the files you have written so far inside the task root.",
    promptSnippet: "List your task files",
    parameters: Type.Object({}),
    async execute() {
      const out: string[] = [];
      if (fs.existsSync(TASK_ROOT)) walk(TASK_ROOT, TASK_ROOT, out);
      return { content: [{ type: "text" as const, text: out.sort().join("\n") || "(no files yet)" }], details: { count: out.length } };
    },
  });
}

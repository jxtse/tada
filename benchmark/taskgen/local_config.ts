/** Read settings through the same Python resolver used by the pipeline. */
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export function configuredValue(key: string): string {
  return execFileSync("python3", [path.join(HERE, "local_config.py"), key, "--required"],
    { encoding: "utf8", env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).trim();
}

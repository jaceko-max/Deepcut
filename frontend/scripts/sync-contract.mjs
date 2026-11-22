// Copies ../contract/*.example.json into lib/contract/ so Vercel builds (root dir = frontend)
// never depend on files outside the project. Skips silently if ../contract is absent.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "..", "contract");
const dst = join(here, "..", "lib", "contract");
mkdirSync(dst, { recursive: true });
for (const f of ["job_done.example.json", "job_running.example.json"]) {
  const from = join(src, f);
  if (existsSync(from)) copyFileSync(from, join(dst, f));
}

// File-access confinement for the read-only research server.
// Three roots only; every path is resolved through realpath so symlinks cannot escape.
import fs from "node:fs";
import path from "node:path";

export type Roots = { research: string; jobs: string; codex: string };
export type RootKey = keyof Roots;

export class Denied extends Error {
  status: number;
  constructor(message: string, status = 403) {
    super(message);
    this.status = status;
  }
}

const READABLE = new Set([".md", ".txt", ".json", ".log", ".csv", ".jsonl"]);
// Never serve anything that looks like a credential, whatever root it sits under.
const SECRET_NAME = /(^|[\\/])(\.env[^\\/]*|.*(secret|credential|passw(or)?d|token|id_rsa|\.pem|\.key|\.p12)[^\\/]*)$/i;
export const MAX_BYTES = 2_000_000;

export function resolveInside(root: string, rel: string): string {
  if (typeof rel !== "string" || rel.length === 0 || rel.length > 600) throw new Denied("bad path", 400);
  if (rel.includes("\0")) throw new Denied("bad path", 400);
  if (path.isAbsolute(rel) || /^[a-zA-Z]:/.test(rel)) throw new Denied("absolute paths are not allowed");
  const parts = rel.split(/[\\/]+/);
  if (parts.some((p) => p === ".." || p === ".")) throw new Denied("path traversal is not allowed");
  const realRoot = fs.realpathSync(root);
  const joined = path.join(realRoot, ...parts);
  let real: string;
  try {
    real = fs.realpathSync(joined);
  } catch {
    throw new Denied("not found", 404);
  }
  if (real !== realRoot && !real.startsWith(realRoot + path.sep)) throw new Denied("outside the allowed roots");
  return real;
}

export function assertReadable(file: string): void {
  if (!READABLE.has(path.extname(file).toLowerCase())) throw new Denied("file type not served");
  if (SECRET_NAME.test(file)) throw new Denied("file name looks like a credential");
}

/** Document ids: "r/<rel>" is under docs/research, "c/<name>" under the codex report dir. */
export function resolveDoc(roots: Roots, id: string): { file: string; key: RootKey } {
  const m = /^([rc])\/(.+)$/.exec(id);
  if (!m) throw new Denied("bad document id", 400);
  const key: RootKey = m[1] === "r" ? "research" : "codex";
  const file = resolveInside(roots[key], m[2]);
  assertReadable(file);
  if (!fs.statSync(file).isFile()) throw new Denied("not a file", 404);
  return { file, key };
}

export function readDoc(roots: Roots, id: string): { id: string; text: string; mtime: number; truncated: boolean } {
  const { file } = resolveDoc(roots, id);
  const st = fs.statSync(file);
  const fd = fs.openSync(file, "r");
  try {
    const len = Math.min(st.size, MAX_BYTES);
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, 0);
    return { id, text: buf.toString("utf8"), mtime: st.mtimeMs, truncated: st.size > MAX_BYTES };
  } finally {
    fs.closeSync(fd);
  }
}

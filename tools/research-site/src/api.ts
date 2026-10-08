import { useCallback, useEffect, useRef, useState } from "react";

type Listener = (topic: string) => void;
const listeners = new Set<Listener>();
const statusListeners = new Set<(up: boolean) => void>();
let source: EventSource | null = null;
let connected = false;

function ensureSource() {
  if (source || typeof EventSource === "undefined") return;
  source = new EventSource("/api/events");
  source.onopen = () => {
    connected = true;
    statusListeners.forEach((f) => f(true));
  };
  source.onerror = () => {
    connected = false;
    statusListeners.forEach((f) => f(false));
  };
  source.onmessage = (e) => {
    try {
      const { topic } = JSON.parse(e.data) as { topic: string };
      listeners.forEach((f) => f(topic));
    } catch {}
  };
}

export function useLiveStatus(): boolean {
  const [up, setUp] = useState(connected);
  useEffect(() => {
    ensureSource();
    statusListeners.add(setUp);
    return () => void statusListeners.delete(setUp);
  }, []);
  return up;
}

export type Api<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  updatedAt: number | null;
  reload: () => void;
};

/** Fetch JSON and refetch when the server reports a change in one of `topics` (or on each 30s tick). */
export function useApi<T>(path: string | null, topics: string[] = []): Api<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(path !== null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const seq = useRef(0);
  const key = topics.join(",");
  const load = useCallback(async () => {
    if (path === null) return;
    const n = ++seq.current;
    try {
      const r = await fetch(path);
      const body = await r.json().catch(() => null);
      if (n !== seq.current) return;
      if (!r.ok) throw new Error((body && body.error) || `HTTP ${r.status}`);
      setData(body as T);
      setError(null);
      setUpdatedAt(Date.now());
    } catch (e) {
      if (n === seq.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (n === seq.current) setLoading(false);
    }
  }, [path]);
  useEffect(() => {
    setData(null);
    setError(null);
    setLoading(path !== null);
    void load();
  }, [load, path]);
  useEffect(() => {
    ensureSource();
    const want = new Set(key ? key.split(",") : []);
    const fn: Listener = (t) => {
      if (t === "tick" || want.has(t)) void load();
    };
    listeners.add(fn);
    return () => void listeners.delete(fn);
  }, [key, load]);
  return { data, error, loading, updatedAt, reload: () => void load() };
}

export type Meta = { researchRoot: string; codexRoot: string; now: number; hasResearch: boolean };
export type IndexDoc = { id: string; text: string; mtime: number; truncated: boolean; stamp: number | null; ageMin: number | null; stale: boolean };
export type Line = {
  branch: string;
  ahead: string;
  behind: string;
  status: "merged" | "open" | "ready" | "ready-old";
  ready_sha: string;
  head_sha: string;
  worker_live: boolean;
  registered: string[];
  gpuq_running: number;
  gpuq_pending: number;
  commit_ts: number | null;
  commit_age: string;
  commit_subject: string;
  report_head: string;
  report_id: string;
  handoff: string;
};
export type Lines = { ok: boolean; error?: string; generated?: number; lines: Line[] };
export type Decision = { date: string; decision: string; quote: string; source: string; section: string; sectionTitle: string; superseded: boolean };
export type DocMeta = { id: string; title: string; mtime: number; size: number };
export type Doc = { id: string; title: string; text: string; mtime: number; truncated: boolean };
export type Hit = { id: string; title: string; snippet: string; score: number };
export type JobRow = {
  id: string;
  label: string;
  line: string;
  state: string;
  display: string;
  priority: number;
  short: boolean;
  gate: boolean;
  memGb: number | null;
  waitS?: number | null;
  runningS?: number | null;
};
export type Stats = { windowH: number; jobs: number; gpuMin: number; wastedMin: number; perLine: { line: string; jobs: number; bad: number; gpuMin: number; wastedMin: number }[] };
export type Gpuq = {
  now: number;
  running: JobRow[];
  pending: JobRow[];
  byPriority: Record<string, number>;
  byLine: Record<string, { pending: number; running: number; maxWaitS: number }>;
  stats1h: Stats;
  stats24h: Stats;
};
export type H2H = Record<string, Record<string, { win: number; tie: number; loss: number; n: number; provisional: boolean }>>;
export type Gap = { model: string; ctx: number; kind: string; metric: string; higherIsBetter: boolean; best_engine: string; best: number; ours: number; ratio: number; reps: Record<string, number> };
export type Parity = {
  mtime: number;
  verdict: string;
  summary: string;
  policy: string;
  parity: number;
  total: number;
  withData: number;
  bestOurs: number;
  gateOpen: boolean;
  missing: number | null;
  headToHead: H2H;
  rejected: { cell: string; job: string; reason: string }[];
  methodFlagged: number;
  engines: string[];
  gaps: Gap[];
};

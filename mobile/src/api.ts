import AsyncStorage from '@react-native-async-storage/async-storage';

import offline from './generated/offline.json';
import { checkCrisis, refreshLexicon } from './safetyValve';

// Set EXPO_PUBLIC_API_URL in mobile/.env (Android emulator: http://10.0.2.2:8000).
export const API_BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';
const TIMEOUT_MS = 4000;
const REF_KEY = 'student_ref.v1';
const PENDING_KEY = 'pending_escalations.v1';

export type Hotline = { name: string; number: string };
export type TriageResponse = {
  crisis: boolean;
  category: string | null;
  responseText: string;
  responseId: string;
  hotlines: Hotline[];
  offline: boolean;
  roundTripMs: number;
};

/** Pseudonymous id generated on the device; lets a counsellor follow up without a name. */
export async function studentRef(): Promise<string> {
  const existing = await AsyncStorage.getItem(REF_KEY);
  if (existing) return existing;
  const ref =
    'anon-' +
    Array.from({ length: 16 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
  await AsyncStorage.setItem(REF_KEY, ref);
  return ref;
}

async function post(text: string, language: 'en' | 'sw', ref: string) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/v1/checkin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language, student_ref: ref }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/** Retry crisis check-ins captured while offline so a counsellor still gets them. */
export async function flushPending(): Promise<void> {
  const raw = await AsyncStorage.getItem(PENDING_KEY);
  if (!raw) return;
  const items: { text: string; language: 'en' | 'sw' }[] = JSON.parse(raw);
  const ref = await studentRef();
  const left = [];
  for (const it of items) {
    try {
      await post(it.text, it.language, ref);
    } catch {
      left.push(it);
    }
  }
  if (left.length) await AsyncStorage.setItem(PENDING_KEY, JSON.stringify(left));
  else await AsyncStorage.removeItem(PENDING_KEY);
}

export async function checkIn(text: string, language: 'en' | 'sw'): Promise<TriageResponse> {
  const t0 = Date.now();
  const ref = await studentRef();
  try {
    const r = await post(text, language, ref);
    const roundTripMs = Date.now() - t0;
    refreshLexicon(API_BASE);
    flushPending();
    return {
      crisis: r.crisis,
      category: r.category,
      responseText: r.response_text,
      responseId: r.response_id,
      hotlines: r.hotlines ?? [],
      offline: false,
      roundTripMs,
    };
  } catch {
    // ---- offline fallback: on-device safety valve -------------------------------
    const v = checkCrisis(text);
    const lang = language === 'sw' ? 'sw' : 'en';
    if (v.hit) {
      const raw = await AsyncStorage.getItem(PENDING_KEY);
      const pending = raw ? JSON.parse(raw) : [];
      pending.push({ text, language });
      await AsyncStorage.setItem(PENDING_KEY, JSON.stringify(pending));
      return {
        crisis: true,
        category: null,
        responseText: offline.crisis[lang],
        responseId: offline.crisis.id,
        hotlines: offline.crisis.hotlines,
        offline: true,
        roundTripMs: Date.now() - t0,
      };
    }
    return {
      crisis: false,
      category: null,
      responseText: offline.fallback[lang],
      responseId: offline.fallback.id,
      hotlines: [],
      offline: true,
      roundTripMs: Date.now() - t0,
    };
  }
}

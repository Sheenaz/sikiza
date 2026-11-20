// Offline safety valve: the same crisis-tier regexes as the server (exported from
// data/lexicon/sheng_lexicon.yaml). Used when the API is unreachable, so a crisis
// message still gets the crisis response and hotlines without a network.
import AsyncStorage from '@react-native-async-storage/async-storage';

import bundled from './generated/lexicon.json';
import { normalise } from './normalise';

type Rule = { id: string; tier: 'crisis' | 'distress'; pattern: string };
type Lexicon = { version: string; rules: Rule[] };

const CACHE_KEY = 'lexicon.v1';
let compiled: { version: string; rules: { id: string; tier: string; re: RegExp }[] } | null = null;

function compile(lex: Lexicon) {
  compiled = {
    version: lex.version,
    rules: lex.rules.flatMap((r) => {
      try {
        return [{ id: r.id, tier: r.tier, re: new RegExp(r.pattern) }];
      } catch {
        return []; // a pattern JS cannot parse is skipped, never crashes the app
      }
    }),
  };
}

export async function loadLexicon(): Promise<void> {
  try {
    const cached = await AsyncStorage.getItem(CACHE_KEY);
    compile(cached ? (JSON.parse(cached) as Lexicon) : (bundled as Lexicon));
  } catch {
    compile(bundled as Lexicon);
  }
}

/** Called after a successful online request; keeps the offline list fresh. */
export async function refreshLexicon(apiBase: string): Promise<void> {
  try {
    const res = await fetch(`${apiBase}/v1/lexicon`);
    if (!res.ok) return;
    const lex = (await res.json()) as Lexicon;
    if (lex.version !== compiled?.version) {
      await AsyncStorage.setItem(CACHE_KEY, JSON.stringify(lex));
      compile(lex);
    }
  } catch {
    /* offline: keep current list */
  }
}

export function checkCrisis(text: string): { hit: boolean; rules: string[] } {
  if (!compiled) compile(bundled as Lexicon);
  const t = normalise(text);
  const rules = compiled!.rules.filter((r) => r.tier === 'crisis' && r.re.test(t)).map((r) => r.id);
  return { hit: rules.length > 0, rules };
}

export function lexiconVersion(): string {
  return compiled?.version ?? (bundled as Lexicon).version;
}

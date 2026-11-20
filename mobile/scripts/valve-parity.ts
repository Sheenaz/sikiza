// Checks the on-device valve makes the same crisis decisions as the Python server.
// Usage: python scripts/valve_cases.py > /tmp/cases.json && node mobile/scripts/valve-parity.ts /tmp/cases.json
import { readFileSync } from 'node:fs';
import { normalise } from '../src/normalise.ts';

const lex = JSON.parse(readFileSync(new URL('../src/generated/lexicon.json', import.meta.url), 'utf8'));
const cases: { text: string; norm: string; rules: string[] }[] = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const rules = lex.rules.filter((r: any) => r.tier === 'crisis').map((r: any) => ({ id: r.id, re: new RegExp(r.pattern) }));
let bad = 0;
for (const c of cases) {
  const n = normalise(c.text);
  const got = rules.filter((r: any) => r.re.test(n)).map((r: any) => r.id);
  if (n !== c.norm || got.join() !== c.rules.join()) {
    bad++;
    console.log('MISMATCH', JSON.stringify({ text: c.text, py: c.norm, js: n, pyRules: c.rules, jsRules: got }));
  }
}
console.log(`${cases.length - bad}/${cases.length} identical`);
process.exit(bad ? 1 : 0);

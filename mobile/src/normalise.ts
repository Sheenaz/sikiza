// Port of triage/text.py::normalise — MUST stay identical so the offline valve
// matches exactly what the server matches. Covered by the parity check in README.

const LEET: Record<string, string> = { '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't' };

export function normalise(text: string): string {
  let t = text.normalize('NFKD').replace(/[̀-ͯ]/g, '');
  t = t.toLowerCase();
  // no lookbehind / \p{L}: keeps the regexes portable across JS engines (Hermes)
  t = t.replace(/(\w)[-_'’](?=\w)/g, '$1');
  t = t.replace(/(\w)\1{2,}/g, '$1$1');
  const words = t.split(/\s+/).filter(Boolean).map((w) => {
    const letters = [...w].filter((c) => /[a-z]/.test(c)).length;
    if (letters && letters >= w.length / 2) {
      return w.replace(/[013457]/g, (d) => LEET[d]);
    }
    return w;
  });
  return words.join(' ').trim();
}

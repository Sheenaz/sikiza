"""Monthly drift check: is new Sheng appearing that the model has not seen?

Input: a month of consented, pseudonymised check-ins (CSV with a `text` column)
plus the training texts. Steps:
  1. embed both sets (multilingual encoder mean-pooled, or TF-IDF+SVD fallback);
  2. novelty = distance to nearest training message, flagged if above the 95th
     percentile of train-to-train nearest-neighbour distances;
  3. cluster the novel messages (KMeans) and list each cluster's most frequent
     tokens that are absent from the training vocabulary and the lexicon —
     candidate new Sheng terms for the counsellor/student panel to review;
  4. report the centroid shift (cosine) between training and the new month.
Output never contains full messages, only aggregate counts and candidate tokens.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .text import normalise, tokens


def tfidf_embedder(train_texts, dim: int = 128):
    vec = TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    x = vec.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=min(dim, x.shape[1] - 1, max(2, len(train_texts) - 1)), random_state=0)
    svd.fit(x)
    return lambda texts: normalize(svd.transform(vec.transform(texts)))


def torch_embedder(bundle_dir):
    from .transformer import TorchScorer, _encode
    import torch

    s = TorchScorer.load(bundle_dir)

    @torch.no_grad()
    def emb(texts):
        out = []
        for i in range(0, len(texts), 64):
            ids, mask = _encode(s.tok, texts[i:i + 64], s.max_length, s.device)
            out.append(s.model.pooled(ids, mask).cpu().numpy())
        return normalize(np.concatenate(out))

    return emb


def drift_report(train_texts: list[str], new_texts: list[str], lexicon_terms: set[str] | None = None,
                 embed=None, n_clusters: int = 8, min_token_count: int = 3) -> dict:
    embed = embed or tfidf_embedder(train_texts)
    a, b = embed(train_texts), embed(new_texts)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(a)
    base = nn.kneighbors(a)[0][:, 1]          # skip self
    # Two rules, each calibrated so that ~2.5% of NO-drift messages trip it (~5% combined).
    cutoff = float(np.quantile(base, 0.975))
    d_new = nn.kneighbors(b, n_neighbors=1)[0][:, 0]
    train_vocab = {t for x in train_texts for t in tokens(x)}

    def oov_of(texts, vocab):
        return np.array([sum(t not in vocab for t in tokens(x)) / max(1, len(tokens(x))) for x in texts])

    # An embedding fit on training text cannot "see" n-grams it never saw (TF-IDF/SVD maps them
    # to ~0), so a message is also novel if an unusually large share of its words is unseen.
    # "Unusually large" is calibrated by 5-fold cross-validation on the training messages: real
    # student text is short and full of typos, so a fixed cut-off (it was 25%) raised false alarms.
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(train_texts))
    ref = np.concatenate([
        oov_of([t for t, f in zip(train_texts, fold) if f == k],
               {w for t, f in zip(train_texts, fold) if f != k for w in tokens(t)})
        for k in range(5)])
    oov_cutoff = float(max(np.quantile(ref, 0.975), 0.25))
    oov_rate = oov_of(new_texts, train_vocab)
    novel = (d_new > cutoff) | (oov_rate > oov_cutoff)

    lex = lexicon_terms or set()
    clusters = []
    if novel.sum() >= 2:
        k = int(min(n_clusters, novel.sum()))
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(b[novel])
        nov_texts = [t for t, f in zip(new_texts, novel) if f]
        for c in range(k):
            cnt = Counter(t for x, l in zip(nov_texts, lab) if l == c for t in set(tokens(x))
                          if t not in train_vocab and t not in lex and t.isalpha() and len(t) > 2)
            clusters.append({
                "cluster": c, "size": int((lab == c).sum()),
                "candidate_new_terms": [(t, n) for t, n in cnt.most_common(10) if n >= min_token_count],
            })
    ca, cb = a.mean(0), b.mean(0)
    shift = 1 - float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
    oov = Counter(t for x in new_texts for t in set(tokens(x)) if t not in train_vocab and t.isalpha())
    return {
        "n_train": len(train_texts), "n_new": len(new_texts),
        "novelty_cutoff": cutoff, "oov_cutoff": oov_cutoff, "novel_share": float(novel.mean()),
        "expected_novel_share_if_no_drift": 0.05,
        "centroid_cosine_shift": shift,
        "oov_token_rate": float(oov_rate.mean()),
        "top_oov_tokens": [(t, n) for t, n in oov.most_common(25) if n >= min_token_count],
        "clusters": clusters,
        "alert": bool(novel.mean() > 0.10),
    }

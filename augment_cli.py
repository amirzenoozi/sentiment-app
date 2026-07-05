import argparse
import os
import re
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from halo import Halo

# Initialize device globally
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# IMPORTANT: run this on the TRAINING SPLIT ONLY, after you have split off your
# validation/test sets. Augmenting the full CSV and splitting afterwards leaks
# paraphrases of the same review into both train and val, which inflates metrics.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lightweight Dutch polarity guard
#
# Back-translation (nl -> pivot -> nl) is lossy exactly where it hurts a
# sentiment task: negation, sarcasm and intensity. A synthetic "Negative"
# review that comes back reading Positive is label noise injected into the
# class we can least afford to corrupt. This heuristic lexicon rejects
# candidates whose surface polarity no longer matches the source label.
#
# It is deliberately a *cheap* guard, not a classifier. For production use a
# model-based filter (e.g. score each candidate with the trained model and keep
# only same-label predictions). Kept self-contained here so the CLI has no
# dependency on a trained checkpoint.
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    "goed", "geweldig", "mooi", "prachtig", "uitstekend", "fantastisch",
    "leuk", "best", "beste", "aanrader", "top", "briljant", "sterk",
    "indrukwekkend", "heerlijk", "perfect", "genieten", "genoten", "boeiend",
    "grappig", "meeslepend", "vermakelijk", "aangenaam", "knap",
}
NEGATIVE_WORDS = {
    "slecht", "saai", "verschrikkelijk", "teleurstellend", "zwak",
    "waardeloos", "vervelend", "matig", "afschuwelijk", "flauw", "traag",
    "voorspelbaar", "belachelijk", "vreselijk", "irritant", "tegenvallend",
    "zonde", "oppervlakkig", "langdradig", "rommelig", "clichématig",
}
NEGATIONS = {"niet", "geen", "nooit", "zonder", "nauwelijks", "amper"}

_TOKEN_RE = re.compile(r"[a-zà-ÿ]+", re.IGNORECASE)


def _word_polarity(tok: str) -> int:
    """Polarity of a single token, tolerant of Dutch adjective inflection.

    Dutch adjectives inflect with -e/-er/-st(e) ("slecht" -> "slechte",
    "slechtste"), so exact matching alone misses most surface forms. We also
    test a few suffix-stripped stems against the base lexicon.
    """
    candidates = {tok}
    for suffix in ("ste", "st", "er", "e"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            candidates.add(tok[: -len(suffix)])
    if candidates & POSITIVE_WORDS:
        return 1
    if candidates & NEGATIVE_WORDS:
        return -1
    return 0


def polarity_score(text: str) -> int:
    """Net polarity of a Dutch text: positive words +1, negative -1.

    A sentiment word is flipped when a negation appears within the previous
    three tokens ("niet goed" -> negative), which is the single most common way
    back-translation silently changes a review's meaning.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    score = 0
    for i, tok in enumerate(tokens):
        base = _word_polarity(tok)
        if base == 0:
            continue
        window = tokens[max(0, i - 3):i]
        if any(w in NEGATIONS for w in window):
            base = -base
        score += base
    return score


def passes_polarity(text: str, direction: str) -> bool:
    """Whether a candidate's surface polarity is consistent with its label.

    direction is 'pos', 'neg' or 'neu'. Neutral requires a near-zero score so
    strongly polarized paraphrases don't drift into the Average class.
    """
    score = polarity_score(text)
    if direction == "pos":
        return score > 0
    if direction == "neg":
        return score < 0
    if direction == "neu":
        return abs(score) <= 1
    return True  # unknown direction -> don't filter


def infer_direction(label, negative_label, positive_label, neutral_label):
    """Map a raw label value to a polarity direction ('pos'/'neg'/'neu'/None).

    Explicit --*_label args win. Otherwise fall back to sensible defaults: the
    training convention (0=negative, 1=average, 2=positive) for integer labels,
    or substring matching for string labels.
    """
    if negative_label is not None and str(label) == negative_label:
        return "neg"
    if positive_label is not None and str(label) == positive_label:
        return "pos"
    if neutral_label is not None and str(label) == neutral_label:
        return "neu"

    s = str(label).strip().lower()
    if s in {"0", "neg", "negative", "negatief"}:
        return "neg"
    if s in {"2", "pos", "positive", "positief"}:
        return "pos"
    if s in {"1", "neu", "neutral", "average", "gemiddeld", "neutraal"}:
        return "neu"
    return None


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
_PIPELINE_CACHE = {}


def get_translation_pipeline(src_lang: str, tgt_lang: str):
    """Load (and cache) the MarianMT tokenizer + model for a language pair."""
    key = (src_lang, tgt_lang)
    if key not in _PIPELINE_CACHE:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        _PIPELINE_CACHE[key] = (tokenizer, model)
    return _PIPELINE_CACHE[key]


def translate_batch(
    texts: list[str], tokenizer, model, batch_size: int = 16,
    do_sample: bool = True, top_p: float = 0.9, temperature: float = 1.0,
    desc: str = "translating", spinner=None, base_text: str = "",
) -> list[str]:
    """Translate texts in batches.

    Sampling (do_sample) is what gives us *diversity*: greedy/beam decoding is
    deterministic, so re-translating the same source produces identical output
    and every cycle after the first yields only duplicates. Sampling makes each
    pass produce fresh variants, so a small minority class can still generate
    enough unique paraphrases.

    If a Halo `spinner` is passed, its text is refreshed each batch so the user
    sees live progress through the current translation pass.
    """
    out = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for b, i in enumerate(range(0, len(texts), batch_size), start=1):
            if spinner is not None:
                spinner.text = f"{base_text} | {desc} batch {b}/{total_batches}"
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            tokens = model.generate(
                **inputs,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=256,
            )
            out.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))
    return out


def back_translate(texts, pivot_lang, batch_size, top_p, temperature,
                   spinner=None, base_text=""):
    """One nl -> pivot -> nl round trip, returning Dutch paraphrases."""
    nl2p_tok, nl2p_mod = get_translation_pipeline("nl", pivot_lang)
    p2nl_tok, p2nl_mod = get_translation_pipeline(pivot_lang, "nl")
    pivot_texts = translate_batch(
        texts, nl2p_tok, nl2p_mod, batch_size, top_p=top_p, temperature=temperature,
        desc=f"nl->{pivot_lang}", spinner=spinner, base_text=base_text,
    )
    return translate_batch(
        pivot_texts, p2nl_tok, p2nl_mod, batch_size, top_p=top_p, temperature=temperature,
        desc=f"{pivot_lang}->nl", spinner=spinner, base_text=base_text,
    )


def generate_for_class(base_texts, needed, direction, label, args):
    """Generate up to `needed` unique, polarity-consistent paraphrases.

    Rotates through the pivot languages and keeps sampling until the target is
    met or the models saturate (few new uniques survive across cycles). A Halo
    spinner shows live status (kept-so-far, current cycle, translation pass).
    """
    pivots = [p.strip() for p in args.pivot_langs.split(",") if p.strip()]
    generated = []
    seen = {t.strip().lower() for t in base_texts}
    rejected_dupe = 0
    rejected_polarity = 0

    spinner = Halo(spinner="dots")
    spinner.start()
    cycle = 0
    while len(generated) < needed and cycle < args.max_cycles:
        pivot = pivots[cycle % len(pivots)]
        base_text = (f"Class {label}: {len(generated)}/{needed} kept "
                     f"(cycle {cycle + 1}, pivot {pivot})")
        spinner.text = base_text
        candidates = back_translate(
            base_texts, pivot, args.batch_size, args.top_p, args.temperature,
            spinner=spinner, base_text=base_text,
        )

        added_this_cycle = 0
        for cand in candidates:
            norm = cand.strip().lower()
            if not norm or norm in seen:
                rejected_dupe += 1
                continue
            if direction is not None and not passes_polarity(cand, direction):
                rejected_polarity += 1
                continue
            seen.add(norm)
            generated.append(cand)
            added_this_cycle += 1
            spinner.text = (f"Class {label}: {len(generated)}/{needed} kept "
                            f"(cycle {cycle + 1}, pivot {pivot})")
            if len(generated) >= needed:
                break

        cycle += 1
        # Saturation guard: if a whole cycle barely adds anything, stop instead
        # of spinning (and heating the machine) for near-zero yield.
        if added_this_cycle <= max(1, needed // 50) and cycle >= len(pivots):
            spinner.info(f"Class {label}: models saturated; stopping early.")
            spinner = None
            break

    summary = (f"Class {label}: kept {len(generated)}/{needed} | rejected "
               f"{rejected_dupe} dupes, {rejected_polarity} polarity-inconsistent")
    if spinner is not None:
        (spinner.succeed if generated else spinner.warn)(summary)
    else:
        print(f"    {summary}")
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Multi-class Dutch dataset balancing via filtered back-translation"
    )
    parser.add_argument("--input", required=True, help="Path to the imbalanced input CSV (training split only)")
    parser.add_argument("--output", required=True, help="Path to save the balanced output CSV")
    parser.add_argument("--text_col", default="Reviews", help="Name of the text column")
    parser.add_argument("--label_col", default="Label", help="Name of the label column")
    parser.add_argument("--target_ratio", type=float, default=0.15, help="Minimum share of the final dataset each under-represented class should reach")
    parser.add_argument("--pivot_langs", default="en,de,fr", help="Comma-separated pivot languages for diversity (e.g. en,de,fr)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p (diversity)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (diversity)")
    parser.add_argument("--max_cycles", type=int, default=8, help="Max augmentation cycles per class before saturation stop")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_polarity_filter", action="store_true", help="Disable the polarity-consistency filter (not recommended)")
    parser.add_argument("--negative_label", default=None, help="Raw label value that means Negative (overrides inference)")
    parser.add_argument("--positive_label", default=None, help="Raw label value that means Positive (overrides inference)")
    parser.add_argument("--neutral_label", default=None, help="Raw label value that means Average/Neutral (overrides inference)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not os.path.exists(args.input):
        print(f"[-] Error: Input file '{args.input}' does not exist.")
        return

    print(f"[*] Loading dataset from {args.input}...")
    df = pd.read_csv(args.input)
    total = len(df)
    counts = df[args.label_col].value_counts()
    print("[+] Initial distribution:")
    for label, cnt in counts.items():
        print(f"      {label}: {cnt} ({cnt / total:.1%})")

    # Which classes fall below the target share -> boost those.
    boosted = [label for label, cnt in counts.items() if cnt / total < args.target_ratio]
    if not boosted:
        print(f"[*] Every class already meets the {args.target_ratio:.0%} target. Nothing to do.")
        df.to_csv(args.output, index=False)
        return

    # Solve for the final size T when each boosted class is raised to
    # target_ratio and the others stay fixed:
    #   T = fixed / (1 - k * target_ratio)
    k = len(boosted)
    if k * args.target_ratio >= 1:
        print(f"[-] Error: {k} classes x {args.target_ratio:.0%} target is not "
              f"feasible (>= 100%). Lower --target_ratio.")
        return
    fixed = int(sum(counts[label] for label in counts.index if label not in boosted))
    final_total = fixed / (1 - k * args.target_ratio)
    per_class_target = round(args.target_ratio * final_total)

    synthetic_frames = []
    for label in boosted:
        current = int(counts[label])
        needed = per_class_target - current
        if needed <= 0:
            continue
        direction = None if args.no_polarity_filter else infer_direction(
            label, args.negative_label, args.positive_label, args.neutral_label
        )
        dir_note = direction or "unfiltered"
        print(f"\n[*] Class {label}: {current} -> {per_class_target} "
              f"(+{needed} synthetic, polarity={dir_note})")
        if not args.no_polarity_filter and direction is None:
            print("    [-] Warning: could not infer polarity direction for this "
                  "label; skipping the filter. Pass --negative_label/etc. to enable it.")

        base_texts = df.loc[df[args.label_col] == label, args.text_col].astype(str).tolist()
        generated = generate_for_class(base_texts, needed, direction, label, args)
        if generated:
            synthetic_frames.append(pd.DataFrame({
                args.text_col: generated,
                args.label_col: label,
            }))

    if not synthetic_frames:
        print("\n[-] No synthetic samples were generated.")
        df.to_csv(args.output, index=False)
        return

    balanced_df = pd.concat([df, *synthetic_frames], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    balanced_df.to_csv(args.output, index=False)

    print(f"\n[+] Saved balanced dataset to {args.output}. Final shape: {balanced_df.shape}")
    new_total = len(balanced_df)
    for label, cnt in balanced_df[args.label_col].value_counts().items():
        print(f"      {label}: {cnt} ({cnt / new_total:.1%})")


if __name__ == "__main__":
    main()

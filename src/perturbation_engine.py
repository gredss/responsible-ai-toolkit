"""
Perturbation Engine for IndoBERT Clickbait Detection

Final perturbation design
-------------------------
  Clean  : 0 % words replaced
  Low    : 10 % of words replaced
  Medium : 20 % of words replaced
  High   : 30 % of words replaced

All three levels use the same mechanism: semantic similarity-based word
substitution using the Indonesian thesaurus (victoriasovereigne/tesaurus).

Candidate filtering pipeline per word
--------------------------------------
  1. Look up the word in the thesaurus (direct → reverse-synonym → stemmed).
  2. Keep candidates with the same POS tag.
  3. Exclude the original word and its antonyms.
  4. Score surviving candidates with IndoBERT contextual cosine similarity.
  5. Keep only candidates with cosine in [0.80, 0.95].

The similarity constraint is applied to original_word ↔ candidate_word in
context, NOT to the full headline.  Headline-level similarity is optional
metadata only.
"""

import os
import json
import random
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

from debug_logger import dbg_perturbation_samples, dbg_perturbation_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SIM_MIN = 0.80
SIM_MAX = 0.95

PERTURBATION_INTENSITIES = {
    "low":    0.10,
    "medium": 0.20,
    "high":   0.30,
}

# Matches runs of alphabetic characters (Unicode-aware).  Punctuation and
# numbers are excluded from the perturbation count.
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)


# ---------------------------------------------------------------------------
# WORD TOKENISATION HELPERS
# ---------------------------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    """Return a list of alphabetic word tokens, punctuation excluded."""
    return WORD_PATTERN.findall(text)


def tokenize_word_spans(text: str) -> List[Dict]:
    """
    Return each word with its character start/end positions.

    Example:
        "Bagaimana pengobatannya?" →
        [{"word": "Bagaimana", "start": 0, "end": 9},
         {"word": "pengobatannya", "start": 10, "end": 23}]
    """
    return [
        {"word": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_PATTERN.finditer(text)
    ]


# ---------------------------------------------------------------------------
# INDOPERT CONTEXTUAL SIMILARITY CHECKER
# ---------------------------------------------------------------------------

class SimilarityChecker:
    """
    Computes IndoBERT contextual cosine similarity between a target word span
    in the original sentence and the same span after each candidate replacement.

    The original embedding is computed once per target word; candidate
    embeddings are batched for efficiency.  Embeddings are L2-normalised so
    cosine similarity reduces to a dot product.

    Parameters
    ----------
    model_name : HuggingFace model id for the encoder
    device     : 'cuda', 'cpu', or None (auto-detect)
    batch_size : number of candidate sentences per forward pass
    """

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        logger.info("Loading IndoBERT similarity model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Similarity model loaded on %s  batch_size=%d", self.device, batch_size)

    # ------------------------------------------------------------------
    # Internal embedding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _span_embedding(self, text: str, start: int, end: int) -> Optional[torch.Tensor]:
        """
        L2-normalised mean of hidden states overlapping [start, end) in text.
        Returns None if no tokens fall within the span.
        """
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, add_special_tokens=True, return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0]
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        hidden = self.model(**encoded).last_hidden_state[0]

        selected = [
            hidden[i]
            for i, (tok_start, tok_end) in enumerate(offsets.tolist())
            if tok_start != tok_end and tok_end > start and tok_start < end
        ]
        if not selected:
            return None

        pooled = torch.stack(selected).mean(dim=0)
        return torch.nn.functional.normalize(pooled, p=2, dim=0).detach().cpu()

    @torch.no_grad()
    def _span_embeddings_batch(
        self,
        texts: List[str],
        spans: List[tuple],
    ) -> List[Optional[torch.Tensor]]:
        """
        Batch version of _span_embedding.  texts[i] and spans[i] correspond to
        the same candidate.  Returns one embedding (or None) per candidate.
        """
        if not texts:
            return []

        all_embeddings: List[Optional[torch.Tensor]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]
            batch_spans = spans[batch_start : batch_start + self.batch_size]

            encoded = self.tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=128, padding=True, add_special_tokens=True,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            hidden  = self.model(**encoded).last_hidden_state  # (B, L, H)

            for b_idx, (start, end) in enumerate(batch_spans):
                selected = [
                    hidden[b_idx, tok_i]
                    for tok_i, (tok_start, tok_end) in enumerate(offsets[b_idx].tolist())
                    if tok_start != tok_end and tok_end > start and tok_start < end
                ]
                if not selected:
                    all_embeddings.append(None)
                    continue
                pooled = torch.stack(selected).mean(dim=0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
                all_embeddings.append(pooled.detach().cpu())

        return all_embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        text: str,
        target_start: int,
        target_end: int,
        candidates: List[str],
    ) -> List[Optional[float]]:
        """
        Return the contextual cosine similarity between the original word span
        and each candidate replacement, in the same order as candidates.

        The original embedding is computed once; candidate embeddings are
        computed in batches.  Embeddings are L2-normalised so cosine = dot.
        """
        if not text.strip():
            return [0.0] * len(candidates)

        candidates = [str(c).strip() for c in candidates if str(c).strip()]
        if not candidates:
            return []

        original_embedding = self._span_embedding(text, target_start, target_end)
        if original_embedding is None:
            return [0.0] * len(candidates)

        # Build the sentence for each candidate (replace the span).
        candidate_texts = []
        candidate_spans = []
        for candidate in candidates:
            candidate_text  = text[:target_start] + candidate + text[target_end:]
            candidate_start = target_start
            candidate_end   = target_start + len(candidate)
            candidate_texts.append(candidate_text)
            candidate_spans.append((candidate_start, candidate_end))

        candidate_embeddings = self._span_embeddings_batch(candidate_texts, candidate_spans)

        scores = []
        for emb in candidate_embeddings:
            if emb is None:
                scores.append(0.0)
            else:
                scores.append(float(torch.dot(original_embedding, emb).item()))
        return scores


@lru_cache(maxsize=1)
def get_similarity_checker() -> SimilarityChecker:
    """
    Return the shared SimilarityChecker, creating it on first use.

    The IndoBERT encoder is loaded lazily so that importing this module does
    not download or load any model weights.
    """
    return SimilarityChecker(batch_size=32)


# ---------------------------------------------------------------------------
# INDONESIAN THESAURUS
# ---------------------------------------------------------------------------

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class IndonesianThesaurus:
    """
    Candidate word generator backed by the Indonesian tesaurus JSON
    (victoriasovereigne/tesaurus).

    Lookup strategy (in order):
        1. Direct exact match on the normalised key.
        2. Reverse synonym lookup (synonym → parent lemma).
        3. Stemmed lookup (Sastrawi stemmer).

    Stemming is used ONLY for lookup; replacements always come from the
    thesaurus text, never from the stemmer output.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Thesaurus not found: {self.path}")

        logger.info("Loading thesaurus: %s", self.path)
        with open(self.path, "r", encoding="utf-8") as fh:
            raw_data = json.load(fh)

        # Normalise all keys once at load time.
        self.data = {str(k).strip().lower(): v for k, v in raw_data.items()}
        logger.info("Thesaurus loaded: %d entries", len(self.data))

        # Sastrawi stemmer (created once).
        self.stemmer = StemmerFactory().create_stemmer()

        # Reverse index: synonym → list of parent entries.
        self.reverse_index: Dict[str, List[Dict]] = {}
        for parent_word, entry in self.data.items():
            if not isinstance(entry, dict) or not entry.get("tag"):
                continue
            for synonym in entry.get("sinonim", []):
                key = str(synonym).strip().lower()
                if key:
                    self.reverse_index.setdefault(key, []).append(
                        {"parent_word": parent_word, "entry": entry}
                    )
        logger.info("Reverse index: %d terms", len(self.reverse_index))

        # Stem index: stemmed form → list of parent entries.
        self.stem_index: Dict[str, List[Dict]] = {}
        for parent_word, entry in self.data.items():
            if not isinstance(entry, dict) or not entry.get("tag"):
                continue
            for word in [parent_word] + [str(s).strip().lower() for s in entry.get("sinonim", [])]:
                stem = self._stem(word)
                if stem:
                    self.stem_index.setdefault(stem, []).append(
                        {"parent_word": parent_word, "entry": entry, "source_word": word}
                    )
        logger.info("Stem index: %d stems", len(self.stem_index))

    @lru_cache(maxsize=100_000)
    def _stem(self, word: str) -> str:
        word = str(word).strip().lower()
        if not word:
            return ""
        if " " in word:
            return word  # do not stem multi-word expressions
        return self.stemmer.stem(word)

    @staticmethod
    def _select_entry(matches: List[Dict]) -> Optional[Dict]:
        """Return the first item in matches whose entry has a POS tag."""
        for item in matches:
            entry = item.get("entry")
            if isinstance(entry, dict) and entry.get("tag"):
                return item
        return None

    @lru_cache(maxsize=100_000)
    def get_entry_with_metadata(self, word: str) -> Optional[Dict]:
        """
        Find the thesaurus entry for word using the three-step lookup.

        Returns a dict with keys: entry, parent_word, lookup_source.
        Returns None if no entry is found.
        """
        key = str(word).strip().lower()
        if not key:
            return None

        # 1. Direct lookup.
        entry = self.data.get(key)
        if entry is not None and entry.get("tag"):
            return {"entry": entry, "parent_word": key, "lookup_source": "direct"}

        # 2. Reverse synonym lookup.
        selected = self._select_entry(self.reverse_index.get(key, []))
        if selected is not None:
            return {"entry": selected["entry"], "parent_word": selected["parent_word"],
                    "lookup_source": "reverse_parent"}

        # 3. Stemmed lookup.
        stem = self._stem(key)
        if stem and stem != key:
            selected = self._select_entry(self.stem_index.get(stem, []))
            if selected is not None:
                return {"entry": selected["entry"], "parent_word": selected["parent_word"],
                        "lookup_source": "stemmed"}

        return None

    def diagnose_candidates(self, word: str) -> Dict:
        """
        Return a diagnostic dict describing which synonym candidates survive
        the POS / antonym filters and are ready for cosine scoring.

        Used by SemanticWordSubstitution to build the eligible-word list.
        Does NOT change any perturbation logic.
        """
        result = {
            "word":                       word,
            "lookup_found":               False,
            "lookup_source":              None,
            "parent_word":                None,
            "original_pos":               None,
            "synonyms_total":             0,
            "rejected_original":          0,
            "rejected_antonym":           0,
            "rejected_pos":               0,
            "pos_compatible_candidates":  0,
            "candidates_for_cosine":      [],
        }

        lookup = self.get_entry_with_metadata(word)
        if lookup is None:
            return result

        result["lookup_found"]   = True
        result["lookup_source"]  = lookup["lookup_source"]
        result["parent_word"]    = lookup["parent_word"]

        entry        = lookup["entry"]
        original_pos = entry.get("tag")
        result["original_pos"]    = original_pos
        result["synonyms_total"]  = len(entry.get("sinonim", []))
        original_lower            = str(word).strip().lower()
        antonyms                  = {str(x).strip().lower() for x in entry.get("antonim", [])}

        for raw_candidate in entry.get("sinonim", []):
            candidate = str(raw_candidate).strip()
            if not candidate:
                continue
            cand_lower = candidate.lower()

            if cand_lower == original_lower:
                result["rejected_original"] += 1
                continue
            if cand_lower in antonyms:
                result["rejected_antonym"] += 1
                continue

            # POS check: reject if the candidate has a different POS tag.
            cand_lookup = self.get_entry_with_metadata(candidate)
            if cand_lookup is not None:
                cand_pos = cand_lookup["entry"].get("tag")
                if cand_pos is not None and cand_pos != original_pos:
                    result["rejected_pos"] += 1
                    continue

            result["pos_compatible_candidates"] += 1
            result["candidates_for_cosine"].append(candidate)

        return result


# ---------------------------------------------------------------------------
# SEMANTIC WORD SUBSTITUTION
# ---------------------------------------------------------------------------

class SemanticWordSubstitution:
    """
    Synonym-based word substitution at a specified intensity.

    Intensity controls the fraction of eligible words that are replaced:
        Low    = 10 %
        Medium = 20 %
        High   = 30 %

    A word is eligible if the thesaurus returns at least one candidate with
    IndoBERT contextual cosine in [SIM_MIN, SIM_MAX].  For each selected word
    the candidate with the highest cosine is used.  Original capitalisation is
    preserved.

    Previously this inherited from BasePerturbation (an abstract base class
    with one concrete subclass).  The ABC added no value and has been removed.
    """

    def __init__(
        self,
        thesaurus_path: str,
        random_seed:    int   = 42,
        sim_min:        float = SIM_MIN,
        sim_max:        float = SIM_MAX,
        similarity_checker: Optional[SimilarityChecker] = None,
    ):
        self.thesaurus   = IndonesianThesaurus(thesaurus_path)
        self.random_seed = random_seed
        self.rng         = random.Random(random_seed)
        self.sim_min     = sim_min
        self.sim_max     = sim_max
        # None -> resolve lazily via get_similarity_checker() on first use.
        self._similarity_checker = similarity_checker

    def _get_checker(self) -> SimilarityChecker:
        """Injected checker if provided, otherwise the lazy shared singleton."""
        if self._similarity_checker is not None:
            return self._similarity_checker
        return get_similarity_checker()

    def perturb(self, text: str, intensity: Optional[float] = None) -> str:
        """Return the perturbed text string (no metadata)."""
        return self.perturb_with_metadata(text=text, intensity=intensity)["perturbed_text"]

    def perturb_with_metadata(self, text: str, intensity: Optional[float] = None) -> Dict:
        """
        Apply synonym substitution and return a rich metadata dict.

        The returned dict contains all success-gate flags and diagnostics
        needed by the evaluation engine:
            perturbed_text, perturbation_success, is_same_as_original,
            similarity_in_range, perturbation_in_range, replacements, …

        Parameters
        ----------
        text      : original headline string
        intensity : fraction of words to replace (must be in (0, 1])
        """
        if text is None:
            text = ""
        text = str(text)

        if intensity is None:
            raise ValueError("Intensity must be explicitly supplied.")
        intensity = float(intensity)
        if not (0 < intensity <= 1):
            raise ValueError(f"Invalid intensity: {intensity}")

        if not text.strip():
            return self._infeasible_result(text, "EMPTY_TEXT", intensity)

        word_spans = tokenize_word_spans(text)
        words      = [item["word"] for item in word_spans]
        if not words:
            return self._infeasible_result(text, "NO_WORDS", intensity)

        # ------------------------------------------------------------------
        # Build the eligible-word list.
        # For each word: look it up in the thesaurus, filter candidates by
        # POS / antonym, then score survivors with IndoBERT cosine.
        # ------------------------------------------------------------------
        lookup_stats      = {"direct": 0, "reverse_parent": 0, "stemmed": 0, "failed": 0}
        diagnostic_stats  = {
            "words_no_dictionary_entry": 0, "words_no_synonyms": 0,
            "candidates_generated": 0,
            "candidates_rejected_original": 0, "candidates_rejected_antonym": 0,
            "candidates_rejected_pos": 0, "candidates_passed_pos": 0,
            "candidates_rejected_cosine_low": 0, "candidates_rejected_cosine_high": 0,
            "candidates_valid": 0,
        }
        word_diagnostics: List[Dict] = []
        eligible:         List[Dict] = []

        for word_pos, word_info in enumerate(word_spans):
            word       = word_info["word"]
            word_start = word_info["start"]
            word_end   = word_info["end"]

            diagnosis              = self.thesaurus.diagnose_candidates(word)
            diagnosis["position"]  = word_pos
            word_diagnostics.append(diagnosis)

            if not diagnosis["lookup_found"]:
                lookup_stats["failed"] += 1
                diagnostic_stats["words_no_dictionary_entry"] += 1
                continue
            if diagnosis["synonyms_total"] == 0:
                diagnostic_stats["words_no_synonyms"] += 1
                continue

            diagnostic_stats["candidates_generated"]          += diagnosis["synonyms_total"]
            diagnostic_stats["candidates_rejected_original"]  += diagnosis["rejected_original"]
            diagnostic_stats["candidates_rejected_antonym"]   += diagnosis["rejected_antonym"]
            diagnostic_stats["candidates_rejected_pos"]       += diagnosis["rejected_pos"]
            diagnostic_stats["candidates_passed_pos"]         += diagnosis["pos_compatible_candidates"]

            candidate_strings = diagnosis["candidates_for_cosine"]
            if not candidate_strings:
                continue

            # Score all POS-compatible candidates in one batched call.
            similarities = self._get_checker().score_candidates(
                text=text,
                target_start=word_start,
                target_end=word_end,
                candidates=candidate_strings,
            )

            valid_candidates: List[Dict] = []
            for candidate, similarity in zip(candidate_strings, similarities):
                if similarity < self.sim_min:
                    diagnostic_stats["candidates_rejected_cosine_low"] += 1
                    continue
                if similarity > self.sim_max:
                    diagnostic_stats["candidates_rejected_cosine_high"] += 1
                    continue
                diagnostic_stats["candidates_valid"] += 1
                valid_candidates.append({
                    "candidate":               candidate,
                    "similarity":              similarity,
                    "candidate_pos":           None,
                    "pos_verified":            True,
                    "pos_source":              "diagnostic_pos_filter",
                    "lookup_source":           diagnosis["lookup_source"],
                    "parent_word":             diagnosis["parent_word"],
                    "original_lookup_source":  diagnosis["lookup_source"],
                    "original_parent_word":    diagnosis["parent_word"],
                })

            if valid_candidates:
                eligible.append({
                    "position":   word_pos,
                    "word":       word,
                    "start":      word_start,
                    "end":        word_end,
                    "candidates": valid_candidates,
                })

        if not eligible:
            return self._infeasible_result(
                text, "NO_VALID_SEMANTIC_CANDIDATE", intensity,
                total_words=len(words), failed_lookup_words=lookup_stats["failed"],
                diagnostic_stats=diagnostic_stats, word_diagnostics=word_diagnostics,
            )

        # ------------------------------------------------------------------
        # Choose how many words to replace.
        # The requested target is never reduced: if too few eligible words
        # exist the perturbation is marked failed (insufficient_eligible_words)
        # but we still replace as many as we can and keep the partial result.
        # ------------------------------------------------------------------
        requested_target = int(np.floor(len(words) * intensity + 0.5))
        insufficient     = len(eligible) < requested_target
        actual_target    = min(requested_target, len(eligible))

        if requested_target == 0:
            return self._infeasible_result(
                text, "TARGET_COUNT_ROUNDED_TO_ZERO", intensity,
                total_words=len(words), eligible_words=len(eligible),
                target_words=requested_target, failed_lookup_words=lookup_stats["failed"],
                diagnostic_stats=diagnostic_stats, word_diagnostics=word_diagnostics,
            )

        # Shuffle eligible words (reproducible via the fixed seed).
        shuffled = eligible.copy()
        self.rng.shuffle(shuffled)

        # Pick actual_target replacements; for each chosen word select the
        # candidate with the highest cosine similarity.
        replacements: List[Dict] = []
        for item in shuffled:
            if len(replacements) >= actual_target:
                break
            best = max(item["candidates"], key=lambda x: x["similarity"])
            replacements.append({
                "position":               item["position"],
                "original":               item["word"],
                "replacement":            best["candidate"],
                "word_cosine_similarity": best["similarity"],
                "candidate_pos":          best["candidate_pos"],
                "pos_verified":           best["pos_verified"],
                "pos_source":             best["pos_source"],
                "lookup_source":          best["lookup_source"],
                "parent_word":            best["parent_word"],
                "original_lookup_source": best["original_lookup_source"],
                "original_parent_word":   best["original_parent_word"],
            })

        if len(replacements) == 0:
            return self._infeasible_result(
                text, "INSUFFICIENT_VALID_REPLACEMENTS", intensity,
                total_words=len(words), eligible_words=len(eligible),
                target_words=requested_target, valid_replacements=0,
                failed_lookup_words=lookup_stats["failed"],
                diagnostic_stats=diagnostic_stats, word_diagnostics=word_diagnostics,
            )

        # ------------------------------------------------------------------
        # Apply substitutions and reconstruct the headline.
        # ------------------------------------------------------------------
        perturbed_words = words.copy()
        for item in replacements:
            replacement = self._preserve_case(item["original"], item["replacement"])
            perturbed_words[item["position"]] = replacement

        perturbed_text = self._replace_words(text, perturbed_words)

        words_changed = sum(
            orig.lower() != repl.lower()
            for orig, repl in zip(words, perturbed_words)
        )
        actual_ratio = words_changed / len(words)

        logger.info(
            "DEBUG replacements=%d requested_target=%d actual_target=%d eligible=%d",
            len(replacements), requested_target, actual_target, len(eligible),
        )

        # ------------------------------------------------------------------
        # Compute success-gate flags.
        # A perturbation succeeds only when ALL FOUR gates hold:
        #   1. The headline actually changed.
        #   2. Every replacement cosine is in [sim_min, sim_max].
        #   3. The word count hit the requested target.
        #   4. There were enough eligible words to meet the target.
        # ------------------------------------------------------------------
        replacement_sims = [r["word_cosine_similarity"] for r in replacements]
        similarity_valid = bool(replacement_sims) and all(
            self.sim_min <= s <= self.sim_max for s in replacement_sims
        )

        sim_mean = float(np.mean(replacement_sims))   if replacement_sims else np.nan
        sim_min  = float(np.min(replacement_sims))    if replacement_sims else np.nan
        sim_max  = float(np.max(replacement_sims))    if replacement_sims else np.nan

        is_same_as_original   = (text == perturbed_text)
        perturbation_in_range = (words_changed == requested_target)
        perturbation_success  = (
            (not is_same_as_original)
            and similarity_valid
            and perturbation_in_range
            and (not insufficient)
        )

        return {
            "original_text":              text,
            "perturbed_text":             perturbed_text,
            "perturbation_level":         None,
            "target_intensity":           intensity,
            "total_words":                len(words),
            "failed_lookup_words":        lookup_stats["failed"],
            "eligible_words":             len(eligible),
            "target_words":               requested_target,
            "actual_target_words":        actual_target,
            "can_satisfy_target":         not insufficient,
            "eligible_shortage":          insufficient,
            "insufficient_eligible_words": insufficient,
            "feasibility_reason":         "INSUFFICIENT_ELIGIBLE_WORDS" if insufficient else "SUFFICIENT_ELIGIBLE_WORDS",
            "words_changed":              words_changed,
            "actual_ratio_all_words":     actual_ratio,
            "diagnostic_stats":           diagnostic_stats,
            "word_diagnostics":           word_diagnostics,
            "is_same_as_original":        is_same_as_original,
            "similarity_in_range":        similarity_valid,
            "perturbation_in_range":      perturbation_in_range,
            "perturbation_success":       perturbation_success,
            "similarity_score_mean":      sim_mean,
            "similarity_score_min":       sim_min,
            "similarity_score_max":       sim_max,
            "replacements":               replacements,
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preserve_case(original: str, replacement: str) -> str:
        """Mirror the capitalisation of the original word onto the replacement."""
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.capitalize()
        return replacement.lower()

    @staticmethod
    def _replace_words(original_text: str, replacement_words: List[str]) -> str:
        """
        Rebuild the headline by replacing word tokens in order while preserving
        all punctuation and spacing from the original.
        """
        it = iter(replacement_words)
        return WORD_PATTERN.sub(lambda m: next(it, m.group(0)), original_text)

    @staticmethod
    def _infeasible_result(
        text:              str,
        reason:            str,
        target_intensity:  float,
        total_words:       int   = 0,
        eligible_words:    int   = 0,
        target_words:      int   = 0,
        valid_replacements: int  = 0,
        failed_lookup_words: int = 0,
        diagnostic_stats:  Optional[Dict] = None,
        word_diagnostics:  Optional[List] = None,
    ) -> Dict:
        """Return a result dict representing an infeasible (failed) perturbation."""
        return {
            "original_text":              text,
            "perturbed_text":             text,
            "perturbation_level":         None,
            "target_intensity":           target_intensity,
            "total_words":                total_words,
            "eligible_words":             eligible_words,
            "target_words":               target_words,
            "words_changed":              valid_replacements,
            "failed_lookup_words":        failed_lookup_words,
            "actual_target_words":        valid_replacements,
            "can_satisfy_target":         eligible_words >= target_words,
            "eligible_shortage":          eligible_words < target_words,
            "insufficient_eligible_words": eligible_words < target_words,
            "feasibility_reason":         reason,
            "diagnostic_stats":           diagnostic_stats if diagnostic_stats is not None else {},
            "word_diagnostics":           word_diagnostics if word_diagnostics is not None else [],
            "actual_ratio_all_words":     valid_replacements / total_words if total_words > 0 else 0.0,
            "is_same_as_original":        True,
            "similarity_in_range":        False,
            "perturbation_in_range":      False,
            "perturbation_success":       False,
            "similarity_score_mean":      np.nan,
            "similarity_score_min":       np.nan,
            "similarity_score_max":       np.nan,
            "replacements":               [],
        }


# ---------------------------------------------------------------------------
# PERTURBATION ENGINE  (public API)
# ---------------------------------------------------------------------------

class PerturbationEngine:
    """
    Public entry point for perturbation.

    Owns a single SemanticWordSubstitution instance and routes calls to it
    based on the requested level (low / medium / high).

    Example
    -------
        engine = PerturbationEngine(thesaurus_path="dataset/data/dict.json")
        perturbed_df = engine.apply_to_dataframe(test_df, level="low")
    """

    def __init__(
        self,
        thesaurus_path: str,
        random_seed:    int   = 42,
        sim_min:        float = SIM_MIN,
        sim_max:        float = SIM_MAX,
        similarity_checker: Optional[SimilarityChecker] = None,
    ):
        self.random_seed  = random_seed
        self.sim_min      = sim_min
        self.sim_max      = sim_max
        self.perturbation = SemanticWordSubstitution(
            thesaurus_path=thesaurus_path,
            random_seed=random_seed,
            sim_min=sim_min,
            sim_max=sim_max,
            similarity_checker=similarity_checker,
        )
        logger.info(
            "PerturbationEngine ready. Levels: low=%.0f%%  medium=%.0f%%  high=%.0f%%  "
            "cosine=[%.2f, %.2f]",
            PERTURBATION_INTENSITIES["low"]    * 100,
            PERTURBATION_INTENSITIES["medium"] * 100,
            PERTURBATION_INTENSITIES["high"]   * 100,
            sim_min, sim_max,
        )

    def apply_perturbation(self, text: str, level: str, intensity: Optional[float] = None) -> str:
        """Return the perturbed text string for the given level (no metadata)."""
        return self.apply_perturbation_with_metadata(text, level, intensity)["perturbed_text"]

    def apply_perturbation_with_metadata(
        self,
        text:      str,
        level:     str,
        intensity: Optional[float] = None,
    ) -> Dict:
        """
        Apply semantic word substitution and return a metadata dict.

        level must be one of 'low', 'medium', 'high'.
        """
        level = level.lower().strip()
        if level not in PERTURBATION_INTENSITIES:
            raise ValueError(
                f"Invalid perturbation level: '{level}'. "
                f"Choose one of: {list(PERTURBATION_INTENSITIES)}."
            )
        if intensity is None:
            intensity = PERTURBATION_INTENSITIES[level]

        result = self.perturbation.perturb_with_metadata(text=text, intensity=intensity)
        result["perturbation_level"] = level
        return result

    def apply_to_dataframe(
        self,
        df:          pd.DataFrame,
        text_column: str            = "text",
        level:       str            = "low",
        intensity:   Optional[float] = None,
        output_csv:  Optional[str]   = None,
    ) -> pd.DataFrame:
        """
        Apply perturbation to every row in df[text_column] and return a new
        DataFrame with the perturbed texts and audit metadata columns.

        Parameters
        ----------
        df          : source DataFrame (a copy is made internally)
        text_column : name of the column containing the original text
        level       : 'low', 'medium', or 'high'
        intensity   : override the default intensity for the level
        output_csv  : if given, save the result DataFrame to this path

        The returned DataFrame keeps all original columns; perturbed texts
        replace df[text_column] and metadata is added as new columns.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

        level = level.lower().strip()
        if level not in PERTURBATION_INTENSITIES:
            raise ValueError(f"Invalid level: '{level}'. Choose one of: {list(PERTURBATION_INTENSITIES)}.")

        logger.info("Applying %s perturbation to %d rows", level.upper(), len(df))

        results = []
        for idx, value in tqdm(df[text_column].items(), total=len(df), desc=f"Perturbing {level.upper()}"):
            text   = "" if pd.isna(value) else str(value)
            result = self.apply_perturbation_with_metadata(text=text, level=level, intensity=intensity)
            results.append({"original_index": idx, **result})

        result_df = pd.DataFrame(results)

        # Re-compute the final success flag from the four gates.
        result_df["perturbation_success"] = (
            (~result_df["is_same_as_original"])
            & result_df["similarity_in_range"]
            & result_df["perturbation_in_range"]
            & (~result_df["insufficient_eligible_words"])
        )

        # Build the output DataFrame from the original, overwriting text_column.
        perturbed_df = df.copy()
        perturbed_df[text_column] = result_df["perturbed_text"].values

        metadata_columns = [
            "original_text", "perturbed_text", "perturbation_level", "target_intensity",
            "failed_lookup_words", "total_words", "eligible_words", "target_words",
            "words_changed", "actual_ratio_all_words",
            "similarity_score_mean", "similarity_score_min", "similarity_score_max",
            "actual_target_words", "is_same_as_original", "similarity_in_range",
            "perturbation_in_range", "insufficient_eligible_words", "eligible_shortage",
            "feasibility_reason", "perturbation_success",
            "diagnostic_stats", "word_diagnostics", "replacements",
        ]
        for col in metadata_columns:
            if col in result_df.columns:
                perturbed_df[col] = result_df[col].values

        # Dataset-level statistics stored in DataFrame.attrs.
        n_success        = int(result_df["perturbation_success"].sum())
        n_total          = len(result_df)
        success_pct      = n_success / n_total * 100 if n_total else 0.0
        perturbation_stats = {
            "mean_word_change_rate": float(result_df["actual_ratio_all_words"].mean()) if n_total else 0.0,
            "mean_similarity_score": float(result_df["similarity_score_mean"].dropna().mean())
                if result_df["similarity_score_mean"].notna().any() else None,
            "min_similarity_score":  float(result_df["similarity_score_min"].dropna().min())
                if result_df["similarity_score_min"].notna().any() else None,
            "max_similarity_score":  float(result_df["similarity_score_max"].dropna().max())
                if result_df["similarity_score_max"].notna().any() else None,
        }
        perturbed_df.attrs["perturbation_stats"] = perturbation_stats

        logger.info("%s success: %d/%d (%.2f%%)", level.upper(), n_success, n_total, success_pct)
        print(f"\n{level.upper()} SUCCESS: {n_success}/{n_total} ({success_pct:.2f}%)")

        dbg_perturbation_samples(
            level=level, domain="dataframe",
            originals=result_df["original_text"].tolist(),
            perturbed=result_df["perturbed_text"].tolist(),
        )
        dbg_perturbation_stats(
            level=level, domain="dataframe", n_texts=len(perturbed_df),
            char_change_mean=0.0,
            word_change_mean=float(result_df["actual_ratio_all_words"].mean()),
        )

        if output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            perturbed_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            logger.info("Saved → %s", output_csv)
            print(f"Saved → {output_csv}")

        return perturbed_df

    def get_perturbation_stats(self, original: str, perturbed: str) -> Dict:
        """Word-level change statistics between original and perturbed headlines."""
        orig_words = tokenize_words(original)
        pert_words = tokenize_words(perturbed)
        if len(orig_words) != len(pert_words):
            raise ValueError("Word substitution should preserve the number of word tokens.")
        changed = sum(a.lower() != b.lower() for a, b in zip(orig_words, pert_words))
        return {
            "original_word_count":  len(orig_words),
            "perturbed_word_count": len(pert_words),
            "words_changed":        changed,
            "word_change_ratio":    changed / len(orig_words) if orig_words else 0.0,
            "is_same_as_original":  original == perturbed,
        }

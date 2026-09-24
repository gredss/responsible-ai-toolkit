# 1 SEP 2349 - Perturbation Engine

"""
Perturbation Engine Module for IndoBERT Clickbait Detection System

Final perturbation design:

    Clean : 0%
    Low   : 10% eligible words
    Medium: 20% eligible words
    High  : 30% eligible words

All three levels use the SAME perturbation method:

    semantic similarity-based word substitution

Candidate generation:
    Indonesian Tesaurus

Candidate filtering:
    1. Same POS
    2. Exclude original word
    3. Exclude antonyms
    4. IndoBERT cosine similarity in [0.80, 0.95]

Important:
    The 0.80-0.95 cosine constraint is applied to:

        original_word <-> replacement_word

    NOT:

        original_headline <-> perturbed_headline

The headline-level similarity is optional metadata only.
"""

import os
import json
import random
import re
import logging

from pathlib import Path
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

# Used only by the character-level typo perturbation (see _TypoSimilarityChecker).
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from debug_logger import (
    dbg_perturbation_samples,
    dbg_perturbation_stats,
)

from tqdm.auto import tqdm

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# FINAL EXPERIMENT CONFIGURATION
# =============================================================================

SIM_MIN = 0.80
SIM_MAX = 0.95

PERTURBATION_INTENSITIES = {
    "low": 0.10,
    "medium": 0.20,
    "high": 0.30,
}


# =============================================================================
# TYPO PERTURBATION CONFIGURATION
# =============================================================================
#
# The typo perturbation is a SEPARATE method from the semantic word
# substitution above. It corrupts characters (keyboard-style typos) instead
# of swapping whole words for synonyms.
#
# It uses its OWN level names so there is never any confusion about what a
# level means:
#
#     "typo_low"    -> edit ~10% of the characters
#     "typo_medium" -> edit ~30% of the characters
#     "typo_high"   -> edit ~50% of the characters
#
# NOTE (important for the thesis write-up):
#   The number here is the TARGET fraction of characters to edit. The typo
#   loop keeps a candidate only if its ACTUAL character edit ratio lands
#   inside [target - tolerance, target + tolerance]. The TF-IDF cosine
#   similarity is ALSO measured for every candidate, but at this stage it is
#   recorded as a DIAGNOSTIC value (with an in-range flag against the
#   0.80-0.95 reference band) and is NOT used to reject candidates. This lets
#   us study empirically how similarity behaves at 10 / 30 / 50 % before
#   deciding whether cosine should become a hard acceptance criterion.

TYPO_INTENSITIES = {
    "typo_low": 0.10,
    "typo_medium": 0.30,
    "typo_high": 0.50,
}

# How far the ACTUAL character edit ratio is allowed to differ from the target
# fraction above and still be accepted (the "little forgiveness"). Example:
# a "typo_medium" target of 0.30 with tolerance 0.05 accepts any candidate
# whose real edit ratio is between 0.25 and 0.35.
TYPO_EDIT_RATIO_TOLERANCE = 0.05

# Reference cosine band used ONLY to set the diagnostic "similarity_in_range"
# flag on typo rows. It does NOT reject candidates (see note above).
TYPO_SIM_REFERENCE_MIN = 0.80
TYPO_SIM_REFERENCE_MAX = 0.95


# =============================================================================
# WORD TOKENIZATION
# =============================================================================

WORD_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+",
    flags=re.UNICODE,
)


def tokenize_words(text: str) -> List[str]:
    """
    Extract alphabetic word tokens.

    Punctuation is excluded from the perturbation count.
    """
    return WORD_PATTERN.findall(text)

def tokenize_word_spans(
    text: str,
) -> List[Dict[str, object]]:
    """
    Return words together with their character positions.

    Example:

        "Bagaimana pengobatannya?"

    returns approximately:

        [
            {
                "word": "Bagaimana",
                "start": 0,
                "end": 9
            },
            {
                "word": "pengobatannya",
                "start": 10,
                "end": 23
            }
        ]
    """

    return [
        {
            "word": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for match in WORD_PATTERN.finditer(text)
    ]

# =============================================================================
# CHARACTER-LEVEL TF-IDF COSINE SIMILARITY (used by the typo perturbation only)
# =============================================================================

class _TypoSimilarityChecker:
    """
    Character-level TF-IDF cosine similarity between two strings.

    This is used ONLY by the typo perturbation. It is a deliberately simple,
    character-based measure (not IndoBERT), because a typo corrupts characters
    rather than swapping meaning-bearing words. It answers the question:
    "how much did the surface text change?" rather than "did the meaning
    change?".

    How it works, step by step:
      1. Each string is turned into character n-grams (2 to 4 characters long)
         using scikit-learn's TfidfVectorizer with analyzer="char_wb".
         "char_wb" means the n-grams are built inside word boundaries, which
         suits short Indonesian headlines.
      2. Each string becomes a TF-IDF vector over those n-grams.
      3. We take the cosine similarity of the two vectors.

    Cosine similarity formula (the same one used throughout this project):

        CosSim(u, v) = (u . v) / (||u|| * ||v||)
                     = sum_i(u_i * v_i)
                       / ( sqrt(sum_i u_i^2) * sqrt(sum_i v_i^2) )

    The result is a float in [0, 1]:
        1.0  -> identical strings
        ~0.0 -> almost no shared character n-grams
    """

    def __init__(self):
        # char_wb = character n-grams that stay within word boundaries.
        # ngram_range=(2, 4) = look at 2-, 3-, and 4-character chunks.
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
        )

    def score(self, original: str, perturbed: str) -> float:
        """
        Return the cosine similarity between two strings, in [0, 1].

        Edge cases are handled explicitly so the caller never crashes:
          - if either string is empty  -> 0.0
          - if the two strings are identical -> 1.0
          - if the vectorizer fails for any reason -> 0.0 (and a warning)
        """
        if not original or not perturbed:
            return 0.0

        if original == perturbed:
            return 1.0

        try:
            # Fit the vectorizer on just these two strings and transform them
            # into TF-IDF vectors. matrix[0] is the original, matrix[1] the
            # perturbed string.
            matrix = self._vectorizer.fit_transform(
                [original, perturbed]
            )

            similarity = cosine_similarity(
                matrix[0],
                matrix[1],
            )[0][0]

            return float(similarity)

        except Exception as exc:
            logger.warning(
                "TF-IDF similarity calculation failed: %s",
                exc,
            )
            return 0.0


# A single shared instance is enough; it holds no per-call state.
_typo_checker = _TypoSimilarityChecker()


# =============================================================================
# INDOBERT CONTEXTUAL SEMANTIC SIMILARITY
# =============================================================================

class _SimilarityChecker:
    """
    IndoBERT contextual similarity checker.

    Optimization:
        1. Original contextual embedding is calculated ONCE per target word.
        2. Candidate sentences are encoded in a batch.
        3. Target-span embeddings are extracted from the batch.
        4. Cosine similarities are calculated against the single
           original embedding.

    Methodology remains unchanged:

        original word in original context
                    VS
        candidate word in modified context

    The target span is represented by the mean of the hidden states
    belonging to the target span.
    """

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        # self.device = "cpu"
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.batch_size = batch_size

        logger.info(
            "Loading IndoBERT contextual similarity model: %s",
            model_name,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        print(f"Model Name: {model_name}")
        print(f"Tokenizer: {self.tokenizer}")
        print(f"Model Used: {self.model}")
        
        self.model.eval()

        logger.info(
            "IndoBERT contextual similarity model loaded on %s",
            self.device,
        )

        logger.info(
            "Similarity inference device: %s",
            self.device,
        )

        logger.info(
            "Similarity batch size: %d",
            self.batch_size,
        )

    # =========================================================================
    # SINGLE TEXT EMBEDDING
    # =========================================================================

    @torch.no_grad()
    def _contextual_span_embedding(
        self,
        text: str,
        start: int,
        end: int,
    ) -> Optional[torch.Tensor]:
        """
        Extract contextual embedding for one target span.

        The full sentence is passed through IndoBERT.

        Target representation =
            mean of hidden states overlapping the target span.

        Output is L2-normalized.
        """

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        offsets = encoded.pop(
            "offset_mapping"
        )[0]

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        output = self.model(**encoded)

        hidden = output.last_hidden_state[0]

        selected_vectors = []

        for token_index, (
            token_start,
            token_end,
        ) in enumerate(offsets.tolist()):

            # Ignore special tokens
            if token_start == token_end:
                continue

            # Token overlaps target character span
            if (
                token_end > start
                and token_start < end
            ):
                selected_vectors.append(
                    hidden[token_index]
                )

        if not selected_vectors:
            return None

        pooled = torch.stack(
            selected_vectors
        ).mean(dim=0)

        pooled = torch.nn.functional.normalize(
            pooled,
            p=2,
            dim=0,
        )

        return pooled.detach().cpu()

    # =========================================================================
    # BATCHED CONTEXTUAL EMBEDDINGS
    # =========================================================================

    @torch.no_grad()
    def _contextual_span_embeddings_batch(
        self,
        texts: List[str],
        spans: List[tuple],
    ) -> List[Optional[torch.Tensor]]:
        """
        Extract contextual span embeddings for multiple texts at once.

        Args:
            texts:
                Candidate sentences.

            spans:
                Character spans corresponding to the candidate
                replacement in each sentence.

        Returns:
            List of normalized span embeddings.

        Example:

            texts = [
                "Pemerintah membuka bursa baru",
                "Pemerintah membuka market baru",
                "Pemerintah membuka pusat baru",
            ]

            spans = [
                (20, 25),
                (20, 26),
                (20, 26),
            ]

        All candidate sentences are processed in one or more
        IndoBERT batches.
        """

        if not texts:
            return []

        all_embeddings = []

        for batch_start in range(
            0,
            len(texts),
            self.batch_size,
        ):

            batch_texts = texts[
                batch_start:
                batch_start + self.batch_size
            ]

            batch_spans = spans[
                batch_start:
                batch_start + self.batch_size
            ]

            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )

            offsets = encoded.pop(
                "offset_mapping"
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            output = self.model(**encoded)

            hidden = output.last_hidden_state

            for batch_index, (
                start,
                end,
            ) in enumerate(batch_spans):

                selected_vectors = []

                token_offsets = offsets[
                    batch_index
                ].tolist()

                for token_index, (
                    token_start,
                    token_end,
                ) in enumerate(token_offsets):

                    # Ignore special tokens / padding
                    if token_start == token_end:
                        continue

                    # Token overlaps target span
                    if (
                        token_end > start
                        and token_start < end
                    ):
                        selected_vectors.append(
                            hidden[
                                batch_index,
                                token_index,
                            ]
                        )

                if not selected_vectors:

                    all_embeddings.append(None)

                    continue

                pooled = torch.stack(
                    selected_vectors
                ).mean(dim=0)

                pooled = torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=0,
                )

                all_embeddings.append(
                    pooled.detach().cpu()
                )

        return all_embeddings

    # =========================================================================
    # SPAN REPLACEMENT
    # =========================================================================

    @staticmethod
    def _replace_span(
        text: str,
        start: int,
        end: int,
        replacement: str,
    ) -> str:
        """
        Replace exact character span.
        """

        return (
            text[:start]
            + replacement
            + text[end:]
        )

    # =========================================================================
    # BATCHED CANDIDATE SCORING
    # =========================================================================

    def score_candidates(
        self,
        text: str,
        target_start: int,
        target_end: int,
        candidates: List[str],
    ) -> List[Optional[float]]:
        """
        Calculate contextual cosine similarity for multiple candidates.

        IMPORTANT:

        Original embedding:
            calculated ONCE.

        Candidate embeddings:
            calculated in batches.

        Returns:
            One cosine similarity per candidate.
        """

        if not text.strip():
            return [
                0.0
                for _ in candidates
            ]

        candidates = [
            str(candidate).strip()
            for candidate in candidates
        ]

        candidates = [
            candidate
            for candidate in candidates
            if candidate
        ]

        if not candidates:
            return []

        # =====================================================================
        # STEP 1
        # Original contextual embedding calculated ONCE
        # =====================================================================

        original_embedding = (
            self._contextual_span_embedding(
                text=text,
                start=target_start,
                end=target_end,
            )
        )

        if original_embedding is None:
            return [
                0.0
                for _ in candidates
            ]

        # =====================================================================
        # STEP 2
        # Build candidate sentences
        # =====================================================================

        candidate_texts = []
        candidate_spans = []

        for candidate in candidates:

            candidate_text = self._replace_span(
                text,
                target_start,
                target_end,
                candidate,
            )

            candidate_start = target_start

            candidate_end = (
                target_start
                + len(candidate)
            )

            candidate_texts.append(
                candidate_text
            )

            candidate_spans.append(
                (
                    candidate_start,
                    candidate_end,
                )
            )

        # =====================================================================
        # STEP 3
        # Candidate embeddings calculated in batches
        # =====================================================================

        candidate_embeddings = (
            self._contextual_span_embeddings_batch(
                texts=candidate_texts,
                spans=candidate_spans,
            )
        )

        # =====================================================================
        # STEP 4
        # Cosine similarity
        #
        # Embeddings are already L2-normalized.
        # Therefore:
        #
        # cosine(a,b) = dot(a,b)
        # =====================================================================

        scores = []

        for candidate_embedding in candidate_embeddings:

            if candidate_embedding is None:

                scores.append(0.0)

                continue

            score = torch.dot(
                original_embedding,
                candidate_embedding,
            )

            scores.append(
                float(score.item())
            )

        return scores

    # =========================================================================
    # BACKWARD-COMPATIBLE SINGLE CANDIDATE METHOD
    # =========================================================================

    def score(
        self,
        text: str,
        target_start: int,
        target_end: int,
        candidate: str,
    ) -> float:
        """
        Single-candidate interface.

        Kept for compatibility.

        For performance-critical code, use score_candidates().
        """

        scores = self.score_candidates(
            text=text,
            target_start=target_start,
            target_end=target_end,
            candidates=[candidate],
        )

        if not scores:
            return 0.0

        return scores[0]


_checker = _SimilarityChecker(
    batch_size=32
)


# =============================================================================
# INDONESIAN THESAURUS
# =============================================================================

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class IndonesianThesaurus:
    """
    Candidate generator using:

        victoriasovereigne/tesaurus

    Lookup strategy:

        1. Direct exact lookup
        2. Reverse synonym -> parent lemma lookup
        3. Stemmed lookup
        4. If no lookup succeeds -> word is not eligible

    Important:
        Stemming is used ONLY for dictionary lookup.

        The actual replacement text always comes from the thesaurus.
        We do NOT replace the original word with its stem.

    Example:

        Sentence:
            "aku mengetahui cara menggunakan abakus"

        "abakus"
            ↓
        reverse synonym lookup
            ↓
        parent = "dekak-dekak"
            ↓
        POS = noun
            ↓
        synonyms:
            cempoa
            sempoa
            swipoa

    For a morphological case:

        "menggunakan"
            ↓
        exact lookup fails
            ↓
        stemmed lookup
            ↓
        matching thesaurus entry
            ↓
        synonyms become candidates
    """

    def __init__(
        self,
        path: str,
    ):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Tesaurus file not found: {self.path}"
            )

        logger.info("START loading thesaurus JSON")
        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as file:
            self.data = json.load(file)

        logger.info(
            "Thesaurus JSON loaded: %d entries",
            len(self.data),
        )

        # ---------------------------------------------------------------------
        # Normalize dictionary keys once
        # ---------------------------------------------------------------------

        self.data = {
            str(key).strip().lower(): value
            for key, value in self.data.items()
        }

        # ---------------------------------------------------------------------
        # Sastrawi stemmer
        # ---------------------------------------------------------------------

        logger.info("START creating Sastrawi stemmer")

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        logger.info("Sastrawi stemmer created")

        # ---------------------------------------------------------------------
        # Reverse synonym index
        #
        # Instead of doing:
        #
        #     for main_word, candidate_entry in self.data.items():
        #
        # for EVERY lookup, build the reverse mapping once.
        #
        # Example:
        #
        #     "abakus" -> [
        #         ("dekak-dekak", entry)
        #     ]
        # ---------------------------------------------------------------------

        self.reverse_index = {}
        logger.info("START building reverse index")
        for parent_word, entry in self.data.items():

            if not isinstance(entry, dict):
                continue

            if not entry.get("tag"):
                continue

            for synonym in entry.get(
                "sinonim",
                [],
            ):

                synonym_key = (
                    str(synonym)
                    .strip()
                    .lower()
                )

                if not synonym_key:
                    continue

                self.reverse_index.setdefault(
                    synonym_key,
                    [],
                ).append(
                    {
                        "parent_word": parent_word,
                        "entry": entry,
                    }
                )

        logger.info(
            "Reverse index complete: %d terms",
            len(self.reverse_index),
        )
        # ---------------------------------------------------------------------
        # Stem index
        #
        # Maps stemmed dictionary words and synonyms to parent entries.
        #
        # Used ONLY after exact lookup fails.
        # ---------------------------------------------------------------------

        self.stem_index = {}
        logger.info("START building stem index")
        for parent_word, entry in self.data.items():

            if not isinstance(entry, dict):
                continue

            if not entry.get("tag"):
                continue

            # Parent lemma itself
            parent_stem = self._stem(
                parent_word
            )

            if parent_stem:
                self.stem_index.setdefault(
                    parent_stem,
                    [],
                ).append(
                    {
                        "parent_word": parent_word,
                        "entry": entry,
                        "source_word": parent_word,
                    }
                )

            # Synonyms
            for synonym in entry.get(
                "sinonim",
                [],
            ):

                synonym_key = (
                    str(synonym)
                    .strip()
                    .lower()
                )

                if not synonym_key:
                    continue

                synonym_stem = self._stem(
                    synonym_key
                )

                if synonym_stem:
                    self.stem_index.setdefault(
                        synonym_stem,
                        [],
                    ).append(
                        {
                            "parent_word": parent_word,
                            "entry": entry,
                            "source_word": synonym_key,
                        }
                    )

        logger.info(
            "Stem index complete: %d stems",
            len(self.stem_index),
        )

        logger.info(
            "Loaded Indonesian thesaurus: %d entries",
            len(self.data),
        )

        logger.info(
            "Built reverse synonym index: %d terms",
            len(self.reverse_index),
        )

        logger.info(
            "Built stem index: %d stems",
            len(self.stem_index),
        )

    # =========================================================================
    # STEMMING
    # =========================================================================

    @lru_cache(maxsize=100000)
    def _stem(
        self,
        word: str,
    ) -> str:
        """
        Stem a word using Sastrawi.

        Cached because the same words appear repeatedly.
        """

        word = str(word).strip().lower()

        if not word:
            return ""

        # Do not stem multi-word expressions as one replacement unit.
        #
        # They will be handled using exact/reverse lookup.
        if " " in word:
            return word

        return self.stemmer.stem(word)

    # =========================================================================
    # INTERNAL ENTRY SELECTION
    # =========================================================================

    @staticmethod
    def _select_entry(
        matches: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        """
        Select first valid parent entry.

        All indexed entries already have POS information.
        """

        for item in matches:

            entry = item.get("entry")

            if (
                isinstance(entry, dict)
                and entry.get("tag")
            ):
                return item

        return None

    # =========================================================================
    # ENTRY LOOKUP
    # =========================================================================

    @lru_cache(maxsize=100000)
    def get_entry_with_metadata(
        self,
        word: str,
    ) -> Optional[Dict[str, object]]:
        """
        Find thesaurus entry.

        Lookup order:

            1. Direct exact entry
            2. Reverse synonym -> parent
            3. Stemmed lookup

        Returns metadata describing how the entry was found.

        Example:

            {
                "entry": {...},
                "parent_word": "dekak-dekak",
                "lookup_source": "reverse_parent"
            }
        """

        key = str(
            word
        ).strip().lower()

        if not key:
            return None

        # =====================================================================
        # 1. DIRECT LOOKUP
        # =====================================================================

        entry = self.data.get(key)

        if (
            entry is not None
            and entry.get("tag")
        ):

            return {
                "entry": entry,
                "parent_word": key,
                "lookup_source": "direct",
            }

        # =====================================================================
        # 2. REVERSE SYNONYM LOOKUP
        #
        # Example:
        #
        #     abakus
        #       ↓
        #     dekak-dekak
        # =====================================================================

        reverse_matches = (
            self.reverse_index.get(
                key,
                [],
            )
        )

        selected = self._select_entry(
            reverse_matches
        )

        if selected is not None:

            return {
                "entry": selected["entry"],
                "parent_word": selected[
                    "parent_word"
                ],
                "lookup_source": "reverse_parent",
            }

        # =====================================================================
        # 3. STEMMED LOOKUP
        #
        # Only used when exact/reverse lookup failed.
        #
        # Example:
        #
        #     menggunakan
        #          ↓
        #     stemmer
        #          ↓
        #     menggunakan -> gunakan
        #          ↓
        #     stem index
        # =====================================================================

        stem = self._stem(key)

        if stem and stem != key:

            stem_matches = (
                self.stem_index.get(
                    stem,
                    [],
                )
            )

            selected = self._select_entry(
                stem_matches
            )

            if selected is not None:

                return {
                    "entry": selected["entry"],
                    "parent_word": selected[
                        "parent_word"
                    ],
                    "lookup_source": "stemmed",
                }

        return None

    # =========================================================================
    # BACKWARD-COMPATIBLE GET ENTRY
    # =========================================================================

    # =========================================================================
    # POS
    # =========================================================================

    # =========================================================================
    # CANDIDATES
    # =========================================================================

    def diagnose_candidates(
        self,
        word: str,
    ) -> Dict[str, object]:
        """
        Diagnose why a word does or does not produce valid candidates.

        This method is for analysis/debugging only.
        It does NOT change the perturbation rules.
        """

        result = {
            "word": word,

            "lookup_found": False,
            "lookup_source": None,
            "parent_word": None,
            "original_pos": None,

            "synonyms_total": 0,

            "rejected_original": 0,
            "rejected_antonym": 0,
            "rejected_pos": 0,

            "pos_compatible_candidates": 0,

            "candidates_for_cosine": [],
        }

        # ================================================================
        # LOOKUP
        # ================================================================

        lookup = self.get_entry_with_metadata(word)

        if lookup is None:
            return result

        result["lookup_found"] = True
        result["lookup_source"] = lookup["lookup_source"]
        result["parent_word"] = lookup["parent_word"]

        entry = lookup["entry"]

        original_word = (
            str(word)
            .strip()
            .lower()
        )

        original_pos = entry.get("tag")

        result["original_pos"] = original_pos

        synonyms = entry.get("sinonim", [])

        result["synonyms_total"] = len(synonyms)

        antonyms = {
            str(x).strip().lower()
            for x in entry.get("antonim", [])
        }

        # ================================================================
        # CANDIDATE FILTERING
        # ================================================================

        for raw_candidate in synonyms:

            candidate = str(
                raw_candidate
            ).strip()

            if not candidate:
                continue

            candidate_lower = candidate.lower()

            # ------------------------------------------------------------
            # Original word
            # ------------------------------------------------------------

            if candidate_lower == original_word:
                result["rejected_original"] += 1
                continue

            # ------------------------------------------------------------
            # Antonym
            # ------------------------------------------------------------

            if candidate_lower in antonyms:
                result["rejected_antonym"] += 1
                continue

            # ------------------------------------------------------------
            # POS
            # ------------------------------------------------------------

            candidate_lookup = (
                self.get_entry_with_metadata(
                    candidate
                )
            )

            if candidate_lookup is not None:

                candidate_entry = (
                    candidate_lookup["entry"]
                )

                candidate_pos = (
                    candidate_entry.get("tag")
                )

                if (
                    candidate_pos is not None
                    and candidate_pos != original_pos
                ):
                    result["rejected_pos"] += 1
                    continue

            # ------------------------------------------------------------
            # Survived dictionary/POS/antonym rules
            # ------------------------------------------------------------

            result[
                "pos_compatible_candidates"
            ] += 1

            result[
                "candidates_for_cosine"
            ].append(candidate)

        return result

# =============================================================================
# BASE PERTURBATION
# =============================================================================

class BasePerturbation(ABC):

    @abstractmethod
    def perturb(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def perturb_with_metadata(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> Dict[str, object]:
        raise NotImplementedError


# =============================================================================
# SEMANTIC WORD SUBSTITUTION
# =============================================================================

class SemanticWordSubstitution(
    BasePerturbation
):
    """
    Single perturbation method used for ALL levels.

    Intensity controls only the percentage of eligible words changed.

        Low    = 10%
        Medium = 20%
        High   = 30%
    """

    def __init__(
        self,
        thesaurus_path: str,
        random_seed: int = 42,
        sim_min: float = SIM_MIN,
        sim_max: float = SIM_MAX,
    ):
        self.thesaurus = IndonesianThesaurus(
            thesaurus_path
        )

        self.random_seed = random_seed

        self.rng = random.Random(
            random_seed
        )

        self.sim_min = sim_min
        self.sim_max = sim_max

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def perturb(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> str:

        result = self.perturb_with_metadata(
            text=text,
            intensity=intensity,
        )

        return result["perturbed_text"]

    # =========================================================================
    # MAIN
    # =========================================================================

    def perturb_with_metadata(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> Dict[str, object]:

        if text is None:
            text = ""

        text = str(text)

        if intensity is None:
            raise ValueError(
                "Intensity must be explicitly supplied."
            )

        intensity = float(
            intensity
        )

        if not (
            0 < intensity <= 1
        ):
            raise ValueError(
                f"Invalid intensity: {intensity}"
            )

        if not text.strip():
            return self._infeasible_result(
                text,
                "EMPTY_TEXT",
                target_intensity=intensity,
            )

        word_spans = tokenize_word_spans(text)

        words = [
            item["word"]
            for item in word_spans
        ]

        if not words:
            return self._infeasible_result(
                text,
                "NO_WORDS",
                target_intensity=intensity,
            )

        # ---------------------------------------------------------------------
        # Determine eligible words
        # ---------------------------------------------------------------------
        lookup_stats = {
            "direct": 0,
            "reverse_parent": 0,
            "stemmed": 0,
            "failed": 0,
        }

        diagnostic_stats = {
            "words_no_dictionary_entry": 0,
            "words_no_synonyms": 0,

            "candidates_generated": 0,

            "candidates_rejected_original": 0,
            "candidates_rejected_antonym": 0,
            "candidates_rejected_pos": 0,

            "candidates_passed_pos": 0,

            "candidates_rejected_cosine_low": 0,
            "candidates_rejected_cosine_high": 0,

            "candidates_valid": 0,
        }

        word_diagnostics = []
        eligible = []
        # candidates = []

        for word_position, word_info in enumerate(word_spans):
            word = word_info["word"]
            word_start = word_info["start"]
            word_end = word_info["end"]

            diagnosis = self.thesaurus.diagnose_candidates(
                word
            )

            diagnosis["position"] = word_position

            word_diagnostics.append(
                diagnosis
            )

            # ================================================================
            # No dictionary entry
            # ================================================================
            if not diagnosis["lookup_found"]:
                lookup_stats["failed"] += 1
                diagnostic_stats[
                    "words_no_dictionary_entry"
                ] += 1
                continue

            # ================================================================
            # No synonym candidates
            # ================================================================
            if diagnosis["synonyms_total"] == 0:
                diagnostic_stats[
                    "words_no_synonyms"
                ] += 1
                continue

            # ================================================================
            # Candidate filtering diagnostics
            # ================================================================
            diagnostic_stats[
                "candidates_generated"
            ] += diagnosis[
                "synonyms_total"
            ]

            diagnostic_stats[
                "candidates_rejected_original"
            ] += diagnosis[
                "rejected_original"
            ]

            diagnostic_stats[
                "candidates_rejected_antonym"
            ] += diagnosis[
                "rejected_antonym"
            ]

            diagnostic_stats[
                "candidates_rejected_pos"
            ] += diagnosis[
                "rejected_pos"
            ]

            diagnostic_stats[
                "candidates_passed_pos"
            ] += diagnosis[
                "pos_compatible_candidates"
            ]

            candidate_strings = diagnosis[
                "candidates_for_cosine"
            ]

            if not candidate_strings:
                continue

            valid_candidates = []

            # =====================================================================
            # ORIGINAL EMBEDDING:
            # calculated ONCE
            #
            # CANDIDATE EMBEDDINGS:
            # calculated in batches
            # =====================================================================
            similarities = _checker.score_candidates(
                text=text,
                target_start=word_start,
                target_end=word_end,
                candidates=candidate_strings,
            )

            for candidate, similarity in zip(
                candidate_strings,
                similarities,
            ):

                if similarity < self.sim_min:
                    diagnostic_stats[
                        "candidates_rejected_cosine_low"
                    ] += 1
                    continue

                if similarity > self.sim_max:
                    diagnostic_stats[
                        "candidates_rejected_cosine_high"
                    ] += 1
                    continue

                diagnostic_stats[
                    "candidates_valid"
                ] += 1

                valid_candidates.append(
                    {
                        "candidate": candidate,
                        "similarity": similarity,

                        "candidate_pos": None,
                        "pos_verified": True,
                        "pos_source": (
                            "diagnostic_pos_filter"
                        ),

                        "lookup_source": (
                            diagnosis["lookup_source"]
                        ),

                        "parent_word": (
                            diagnosis["parent_word"]
                        ),

                        "original_lookup_source": (
                            diagnosis["lookup_source"]
                        ),

                        "original_parent_word": (
                            diagnosis["parent_word"]
                        ),
                    }
                )

            if valid_candidates:
                eligible.append(
                    {
                        "position": word_position,
                        "word": word,
                        "start": word_start,
                        "end": word_end,
                        "candidates": valid_candidates,
                    }
                )

        if not eligible:
            return self._infeasible_result(
                text,
                "NO_VALID_SEMANTIC_CANDIDATE",
                target_intensity=intensity,
                total_words=len(words),
                failed_lookup_words=lookup_stats["failed"],
                diagnostic_stats=diagnostic_stats,
                word_diagnostics=word_diagnostics,
            )

        # ---------------------------------------------------------------------
        # Target number of changed words
        #
        # Intensity is defined against all words in the original text.
        # Only eligible words can be selected to satisfy this target.
        # ---------------------------------------------------------------------

        requested_target_count  = int(
            np.floor(
                len(words) * intensity + 0.5
            )
        )

        target_count = min(
            requested_target_count ,
            len(eligible),
        )

        if target_count == 0:
            return self._infeasible_result(
                text,
                "TARGET_COUNT_ROUNDED_TO_ZERO",
                target_intensity=intensity,
                total_words=len(words),
                eligible_words=len(eligible),
                target_words=target_count,
                valid_replacements=0,
                failed_lookup_words=lookup_stats["failed"],
                diagnostic_stats=diagnostic_stats,
                word_diagnostics=word_diagnostics,
            )


        # ---------------------------------------------------------------------
        # Shuffle eligible words so the selected positions vary while remaining
        # reproducible through the random seed.
        # ---------------------------------------------------------------------

        shuffled = eligible.copy()

        self.rng.shuffle(
            shuffled
        )

        replacements = []

        # ---------------------------------------------------------------------
        # Select enough valid word replacements.
        #
        # If selected word cannot be replaced, move to another eligible word.
        # ---------------------------------------------------------------------

        for item in shuffled:
            if len(replacements) >= target_count:
                break

            candidates = item[
                "candidates"
            ]

            # highest contextual semantic similarity: 0.95
            #best = self.rng.choice(candidates)
            best = max(
                candidates,
                key=lambda x: x["similarity"],
            )

            replacements.append(
                {
                    "position": item["position"],

                    "original": item["word"],

                    "replacement": best[
                        "candidate"
                    ],

                    "word_cosine_similarity": (
                        best["similarity"]
                    ),

                    "candidate_pos": (
                        best["candidate_pos"]
                    ),

                    "pos_verified": (
                        best["pos_verified"]
                    ),

                    "pos_source": (
                        best["pos_source"]
                    ),

                    "lookup_source": (
                        best["lookup_source"]
                    ),

                    "parent_word": (
                        best["parent_word"]
                    ),

                    "original_lookup_source": (
                        best["original_lookup_source"]
                    ),

                    "original_parent_word": (
                        best["original_parent_word"]
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Cannot satisfy requested intensity
        # ---------------------------------------------------------------------

        if len(replacements) < target_count:
            return self._infeasible_result(
                text,
                "INSUFFICIENT_VALID_REPLACEMENTS",
                target_intensity=intensity,
                total_words=len(words),
                eligible_words=len(eligible),
                target_words=target_count,
                valid_replacements=len(replacements),
                failed_lookup_words=lookup_stats["failed"],
                diagnostic_stats=diagnostic_stats,
                word_diagnostics=word_diagnostics,
            )

        # ---------------------------------------------------------------------
        # Apply substitutions
        # ---------------------------------------------------------------------

        perturbed_words = words.copy()

        for item in replacements:

            replacement = item[
                "replacement"
            ]

            original = item[
                "original"
            ]

            replacement = (
                self._preserve_case(
                    original,
                    replacement,
                )
            )

            perturbed_words[
                item["position"]
            ] = replacement

        # ---------------------------------------------------------------------
        # Reconstruct headline
        # ---------------------------------------------------------------------

        perturbed_text = self._replace_words(
            text,
            perturbed_words,
        )

        words_changed = sum(
            original.lower()
            != replacement.lower()
            for original, replacement
            in zip(
                words,
                perturbed_words,
            )
        )

        actual_ratio_all_words = (
            words_changed
            / len(words)
        )

        actual_ratio_eligible = (
            words_changed
            / len(eligible)
        )

        # ---------------------------------------------------------------------
        # Validate every replacement
        # ---------------------------------------------------------------------
        logger.info(
            "DEBUG replacements=%d target_count=%d eligible=%d",
            len(replacements),
            target_count,
            len(eligible),
        )

        replacement_similarities = [
            x["word_cosine_similarity"]
            for x in replacements
        ]

        similarity_valid = (
            bool(replacement_similarities)
            and all(
                self.sim_min <= score <= self.sim_max
                for score in replacement_similarities
            )
        )

        if replacement_similarities:
            similarity_mean = float(np.mean(replacement_similarities))
            similarity_min = float(np.min(replacement_similarities))
            similarity_max = float(np.max(replacement_similarities))
        else:
            similarity_mean = np.nan
            similarity_min = np.nan
            similarity_max = np.nan

        return {
            "original_text": text,
            "perturbed_text": perturbed_text,

            "perturbation_level": None,

            "target_intensity": intensity,

            "total_words": len(words),

            "failed_lookup_words": lookup_stats["failed"],

            "eligible_words": len(eligible),
            "target_words": target_count,

            "can_satisfy_target": (
                len(eligible) >= target_count
            ),
            "eligible_shortage": (
                len(eligible) < target_count
            ),
            "feasibility_reason": (
                "SUFFICIENT_ELIGIBLE_WORDS"
                if len(eligible) >= target_count
                else "INSUFFICIENT_ELIGIBLE_WORDS"
            ),

            "words_changed": words_changed,

            "actual_ratio_all_words": actual_ratio_all_words,

            "diagnostic_stats": diagnostic_stats,
            "word_diagnostics": word_diagnostics,

            "is_same_as_original": text == perturbed_text,

            "similarity_in_range": similarity_valid,

            "perturbation_in_range": (
                words_changed == target_count
            ),

            # Neutral, type-agnostic names: for the semantic mechanism this is
            # the per-word contextual cosine; for typo it is the whole-text
            # character TF-IDF cosine. The name no longer implies "word".
            "similarity_score_mean": similarity_mean,
            "similarity_score_min": similarity_min,
            "similarity_score_max": similarity_max,
            "replacements": replacements,
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _preserve_case(
        original: str,
        replacement: str,
    ) -> str:

        if original.isupper():
            return replacement.upper()

        if original.istitle():
            return replacement.capitalize()

        return replacement.lower()

    @staticmethod
    def _replace_words(
        original_text: str,
        replacement_words: List[str],
    ) -> str:
        """
        Replace word tokens in original text while preserving
        original punctuation and spacing as much as possible.
        """

        replacement_iter = iter(
            replacement_words
        )

        def replace_match(match):
            try:
                return next(
                    replacement_iter
                )
            except StopIteration:
                return match.group(0)

        return WORD_PATTERN.sub(
            replace_match,
            original_text,
        )

    @staticmethod
    def _infeasible_result(
        text: str,
        reason: str,
        target_intensity: float,
        total_words: int = 0,
        eligible_words: int = 0,
        target_words: int = 0,
        valid_replacements: int = 0,
        failed_lookup_words: int = 0,
        diagnostic_stats: Optional[Dict[str, object]] = None,
        word_diagnostics: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:

        return {
            "original_text": text,
            "perturbed_text": text,

            "perturbation_level": None,

            "target_intensity": target_intensity,

            "total_words": total_words,

            "eligible_words": eligible_words,

            "target_words": target_words,

            "words_changed": valid_replacements,

            "failed_lookup_words": failed_lookup_words,

            # ============================================================
            # FEASIBILITY
            # ============================================================

            "can_satisfy_target": (
                eligible_words >= target_words
            ),

            "eligible_shortage": (
                eligible_words < target_words
            ),

            "feasibility_reason": reason,

            # ============================================================
            # DIAGNOSTICS
            # ============================================================

            "diagnostic_stats": (
                diagnostic_stats
                if diagnostic_stats is not None
                else {}
            ),

            "word_diagnostics": (
                word_diagnostics
                if word_diagnostics is not None
                else []
            ),

            "actual_ratio_all_words": (
                valid_replacements / total_words
                if total_words > 0
                else 0.0
            ),

            "is_same_as_original": True,

            "similarity_in_range": False,

            "perturbation_in_range": False,

            "similarity_score_mean": np.nan,

            "similarity_score_min": np.nan,

            "similarity_score_max": np.nan,

            "replacements": [],
        }


# =============================================================================
# TYPO (CHARACTER-LEVEL) PERTURBATION
# =============================================================================

class TypoCharacterPerturbation(BasePerturbation):
    """
    Character-level typo perturbation.

    This is a SEPARATE method from SemanticWordSubstitution. Instead of
    swapping whole words for synonyms, it introduces realistic keyboard typos
    into the characters of the text.

    Four typo operations are used (all based on a QWERTY keyboard layout):
        - substitution : replace a letter with a neighbouring key   (e.g. n->m)
        - delete       : drop a letter                              (e.g. buku->buk)
        - insert       : add a neighbouring key after a letter      (e.g. bu->bku)
        - swap         : swap two adjacent letters                  (e.g. ab->ba)

    HOW A LEVEL IS SATISFIED
    ------------------------
    Each level has a TARGET character edit ratio (see TYPO_INTENSITIES):

        typo_low    -> 0.10   (edit ~10% of characters)
        typo_medium -> 0.30   (edit ~30% of characters)
        typo_high   -> 0.50   (edit ~50% of characters)

    The flow for one text is exactly:

        target ratio (0.10 / 0.30 / 0.50)
              |
              v
        generate a candidate (apply typo operations)
              |
              v
        measure the ACTUAL character edit ratio (Levenshtein / length)
              |
              v
        measure the TF-IDF cosine similarity
              |
              v
        record BOTH values
              |
              v
        keep the candidate only if the ACTUAL edit ratio is within
        [target - tolerance, target + tolerance]; otherwise try again.

    IMPORTANT: cosine similarity is measured and stored, but it does NOT
    reject a candidate at this stage. We keep it as a diagnostic so we can
    study, empirically, how similarity behaves at 10 / 30 / 50 % before
    deciding whether to promote it to a hard acceptance criterion.
    """

    def __init__(self, random_seed: int = 42):
        # A private, seeded random generator makes the typos reproducible
        # without touching Python's global random state.
        self._rng = random.Random(random_seed)

        # QWERTY neighbours for each letter. Used to pick a "realistic" wrong
        # key for substitution and insertion.
        self.keyboard_neighbors = {
            "a": ["s", "q", "w", "z"],
            "b": ["v", "g", "h", "n"],
            "c": ["x", "d", "f", "v"],
            "d": ["s", "e", "r", "f", "c", "x"],
            "e": ["w", "r", "d", "s"],
            "f": ["d", "r", "t", "g", "v", "c"],
            "g": ["f", "t", "y", "h", "b", "v"],
            "h": ["g", "y", "u", "j", "n", "b"],
            "i": ["u", "o", "k", "j"],
            "j": ["h", "u", "i", "k", "m", "n"],
            "k": ["j", "i", "o", "l", "m"],
            "l": ["k", "o", "p"],
            "m": ["n", "j", "k"],
            "n": ["b", "h", "j", "m"],
            "o": ["i", "p", "l", "k"],
            "p": ["o", "l"],
            "q": ["w", "a"],
            "r": ["e", "t", "f", "d"],
            "s": ["a", "w", "e", "d", "x", "z"],
            "t": ["r", "y", "g", "f"],
            "u": ["y", "i", "j", "h"],
            "v": ["c", "f", "g", "b"],
            "w": ["q", "e", "s", "a"],
            "x": ["z", "s", "d", "c"],
            "y": ["t", "u", "h", "g"],
            "z": ["a", "s", "x"],
        }

        logger.info("TypoCharacterPerturbation initialized")

    # =========================================================================
    # PUBLIC (text-only, kept for API symmetry with the other perturbation)
    # =========================================================================

    def perturb(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> str:
        result = self.perturb_with_metadata(
            text=text,
            intensity=intensity,
        )
        return result["perturbed_text"]

    # =========================================================================
    # MAIN
    # =========================================================================

    def perturb_with_metadata(
        self,
        text: str,
        intensity: Optional[float] = None,
        max_attempts: int = 60,
    ) -> Dict[str, object]:
        """
        Apply typo perturbation to one text and return full metadata.

        Args:
            text:        the input string.
            intensity:   TARGET fraction of characters to edit (e.g. 0.30).
                         Must be supplied by the caller (PerturbationEngine
                         passes the value that matches the chosen level).
            max_attempts: how many candidates to try while searching for one
                         whose actual edit ratio lands in the target band.

        Returns:
            A metadata dict that reuses the SAME contract keys as the semantic
            word substitution (original_text, perturbed_text, is_same_as_original,
            similarity_in_range, perturbation_in_range, actual_ratio_all_words,
            etc.) so the DataFrame/CSV columns stay consistent, PLUS a few
            typo-specific, honestly-named keys (char_edit_ratio, typo_operations,
            char_cosine_similarity).
        """
        if text is None:
            text = ""

        text = str(text)

        if intensity is None:
            raise ValueError(
                "Typo intensity must be explicitly supplied."
            )

        target_ratio = float(intensity)

        # Empty / whitespace-only text cannot be perturbed.
        if not text.strip():
            return self._infeasible_result(
                text,
                reason="EMPTY_TEXT",
                target_intensity=target_ratio,
            )

        # Count the characters that can actually be typo'd (letters only).
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())

        if alpha_chars == 0:
            return self._infeasible_result(
                text,
                reason="NO_ALPHABETIC_CHARACTERS",
                target_intensity=target_ratio,
                total_chars=total_chars,
            )

        # Accept any candidate whose real edit ratio is within this window.
        lower_bound = target_ratio - TYPO_EDIT_RATIO_TOLERANCE
        upper_bound = target_ratio + TYPO_EDIT_RATIO_TOLERANCE

        # -------------------------------------------------------------------
        # Try up to max_attempts candidates. Keep every candidate we generate
        # so that, if none land inside the band, we can still return the
        # closest one (clearly flagged as out of range) instead of crashing.
        # -------------------------------------------------------------------
        best_candidate = None          # candidate closest to the target band
        best_distance = None           # how far that candidate's ratio is from the band

        for _ in range(max_attempts):

            candidate, operations = self._generate_candidate(
                text,
                target_ratio,
            )

            # Skip candidates that did not actually change anything.
            if candidate == text:
                continue

            # STEP: actual character edit ratio (Levenshtein / length).
            actual_ratio = self._character_edit_ratio(text, candidate)

            # STEP: TF-IDF cosine similarity (diagnostic only).
            similarity = _typo_checker.score(text, candidate)

            # Is the actual edit ratio inside the accepted band?
            ratio_in_band = (lower_bound <= actual_ratio <= upper_bound)

            if ratio_in_band:
                # Found a good candidate: build and return immediately.
                return self._build_result(
                    original=text,
                    perturbed=candidate,
                    operations=operations,
                    actual_ratio=actual_ratio,
                    similarity=similarity,
                    total_chars=total_chars,
                    target_intensity=target_ratio,
                    edit_ratio_in_band=True,
                )

            # Otherwise remember the closest-to-band candidate as a fallback.
            distance = self._distance_to_band(
                actual_ratio,
                lower_bound,
                upper_bound,
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_candidate = (
                    candidate,
                    operations,
                    actual_ratio,
                    similarity,
                )

        # -------------------------------------------------------------------
        # No candidate landed inside the target edit-ratio band.
        # Return the closest one we saw, clearly flagged as out of range.
        # (If we somehow never produced a changed candidate, report infeasible.)
        # -------------------------------------------------------------------
        if best_candidate is None:
            return self._infeasible_result(
                text,
                reason="NO_CANDIDATE_CHANGED_TEXT",
                target_intensity=target_ratio,
                total_chars=total_chars,
            )

        candidate, operations, actual_ratio, similarity = best_candidate

        return self._build_result(
            original=text,
            perturbed=candidate,
            operations=operations,
            actual_ratio=actual_ratio,
            similarity=similarity,
            total_chars=total_chars,
            target_intensity=target_ratio,
            edit_ratio_in_band=False,
        )

    # =========================================================================
    # RESULT BUILDERS
    # =========================================================================

    def _build_result(
        self,
        original: str,
        perturbed: str,
        operations: List[str],
        actual_ratio: float,
        similarity: float,
        total_chars: int,
        target_intensity: float,
        edit_ratio_in_band: bool,
    ) -> Dict[str, object]:
        """
        Assemble the metadata dict for a successfully generated typo candidate.

        The keys are split into two groups:
          1. SHARED CONTRACT KEYS  - the same names the semantic word
             substitution returns, so downstream code and the output CSV keep
             a consistent schema. For a typo the "word" fields do not apply, so
             they are filled with honest, neutral values (0 / NaN / empty).
          2. TYPO-SPECIFIC KEYS    - clearly named so nobody mistakes a
             character measurement for a word-level one.
        """
        # Diagnostic: is the recorded cosine inside the 0.80-0.95 reference
        # band? This is informational only for typos (it does not gate anything).
        similarity_in_reference_band = (
            TYPO_SIM_REFERENCE_MIN <= similarity <= TYPO_SIM_REFERENCE_MAX
        )

        return {
            # ---- SHARED CONTRACT KEYS -------------------------------------
            "original_text": original,
            "perturbed_text": perturbed,

            # Filled in by PerturbationEngine (e.g. "typo_medium").
            "perturbation_level": None,

            "target_intensity": target_intensity,

            # For a typo we count characters, not words. We still populate the
            # shared ratio key so the dataset-level stats keep working, and we
            # define it as the character edit ratio.
            "actual_ratio_all_words": actual_ratio,

            # Word-substitution fields do not apply to a typo. Keep the keys
            # (for a consistent schema) but fill them with neutral values.
            "total_words": 0,
            "eligible_words": 0,
            "target_words": 0,
            "words_changed": 0,
            "failed_lookup_words": 0,
            "diagnostic_stats": {},
            "word_diagnostics": [],
            "replacements": [],

            # These similarity keys exist so the shared stats/logging code can
            # read them. For a typo the meaningful similarity is the character
            # TF-IDF cosine, so we mirror it here (mean == min == max, since a
            # typo produces a single whole-text similarity, not per-word ones).
            "similarity_score_mean": similarity,
            "similarity_score_min": similarity,
            "similarity_score_max": similarity,

            "is_same_as_original": original == perturbed,

            # For a typo, "in range" means the cosine is inside the 0.80-0.95
            # reference band. Recorded as a diagnostic; it does NOT reject.
            "similarity_in_range": similarity_in_reference_band,

            # "perturbation_in_range" means the perturbation hit its own target,
            # which for a typo is the character edit-ratio band.
            "perturbation_in_range": edit_ratio_in_band,

            # ---- TYPO-SPECIFIC KEYS (honestly named) ----------------------
            "char_edit_ratio": actual_ratio,
            "char_cosine_similarity": similarity,
            "total_chars": total_chars,
            "typo_operations": ", ".join(operations) if operations else "",
        }

    def _infeasible_result(
        self,
        text: str,
        reason: str,
        target_intensity: float,
        total_chars: int = 0,
    ) -> Dict[str, object]:
        """
        Result returned when no typo could be applied (empty text, no letters,
        or no candidate ever changed the text). Mirrors the shared schema with
        neutral values so downstream code does not need special-casing.
        """
        return {
            # ---- SHARED CONTRACT KEYS -------------------------------------
            "original_text": text,
            "perturbed_text": text,
            "perturbation_level": None,
            "target_intensity": target_intensity,
            "actual_ratio_all_words": 0.0,
            "total_words": 0,
            "eligible_words": 0,
            "target_words": 0,
            "words_changed": 0,
            "failed_lookup_words": 0,
            "diagnostic_stats": {},
            "word_diagnostics": [],
            "replacements": [],
            "similarity_score_mean": np.nan,
            "similarity_score_min": np.nan,
            "similarity_score_max": np.nan,
            "is_same_as_original": True,
            "similarity_in_range": False,
            "perturbation_in_range": False,
            # ---- TYPO-SPECIFIC KEYS ---------------------------------------
            "char_edit_ratio": 0.0,
            "char_cosine_similarity": np.nan,
            "total_chars": total_chars,
            "typo_operations": "",
            "feasibility_reason": reason,
        }

    # =========================================================================
    # EDIT-RATIO HELPERS
    # =========================================================================

    @staticmethod
    def _distance_to_band(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        """How far a value is from the [lower, upper] band (0 if inside)."""
        if value < lower:
            return lower - value
        if value > upper:
            return value - upper
        return 0.0

    def _character_edit_ratio(
        self,
        original: str,
        perturbed: str,
    ) -> float:
        """
        Actual character edit ratio = Levenshtein distance / alphabetic-char count.

        The denominator is the number of ALPHABETIC characters in the original
        text, to stay consistent with how the target number of edits is chosen
        in _generate_candidate (target_edits = round(alpha_count * intensity)).
        Using total length here (including digits, spaces, punctuation) would
        bias the measured ratio downward for texts with many non-letters and
        make the target band harder to hit.

        This is the HONEST measure of how much changed. We compute it against
        the generated candidate rather than trusting the number of operations
        we asked for, because operations like swap or insert can affect the
        distance differently than a naive count would suggest.
        """
        if not original:
            return 0.0

        alpha_count = sum(1 for c in original if c.isalpha())
        if alpha_count == 0:
            return 0.0

        return self._edit_distance(original, perturbed) / alpha_count

    @staticmethod
    def _edit_distance(original: str, perturbed: str) -> int:
        """
        Standard Levenshtein edit distance (minimum single-character
        insertions, deletions, or substitutions to turn one string into the
        other). Implemented with a simple two-row dynamic-programming table
        so it is easy to read and needs no external library.
        """
        m = len(original)
        n = len(perturbed)

        if m == 0:
            return n
        if n == 0:
            return m

        previous_row = list(range(n + 1))

        for i in range(1, m + 1):
            current_row = [i]
            for j in range(1, n + 1):
                cost_insert = current_row[j - 1] + 1
                cost_delete = previous_row[j] + 1
                cost_substitute = previous_row[j - 1] + (
                    original[i - 1] != perturbed[j - 1]
                )
                current_row.append(
                    min(cost_insert, cost_delete, cost_substitute)
                )
            previous_row = current_row

        return previous_row[n]

    # =========================================================================
    # CANDIDATE GENERATION
    # =========================================================================

    def _generate_candidate(
        self,
        text: str,
        intensity: float,
    ) -> Tuple[str, List[str]]:
        """
        Produce one typo'd version of the text.

        We aim to edit roughly `intensity` fraction of the alphabetic
        characters, choosing a random operation for each edit. The caller
        checks the ACTUAL edit ratio afterwards, so this method only needs to
        get close to the target.

        Returns:
            (candidate_text, list_of_operation_names)
        """
        chars = list(text)

        alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
        alpha_count = len(alpha_indices)

        if alpha_count == 0:
            return text, []

        # How many edits to aim for, based on the target fraction. At least 1.
        target_edits = int(round(alpha_count * intensity))
        target_edits = max(1, target_edits)

        operations_used: List[str] = []
        edits_done = 0

        # Cap the inner loop so we can never spin forever on awkward strings.
        max_inner_attempts = max(50, target_edits * 20)
        inner_attempts = 0

        while edits_done < target_edits and inner_attempts < max_inner_attempts:
            inner_attempts += 1

            # Recompute letter positions each time, because insert/delete
            # change the length and shift positions.
            alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
            if not alpha_indices:
                break

            # If only one edit remains, avoid "swap" because a swap can count
            # as two edits and overshoot the target.
            if target_edits - edits_done == 1:
                operation = self._rng.choice(
                    ["substitution", "delete", "insert"]
                )
            else:
                operation = self._rng.choice(
                    ["substitution", "delete", "insert", "swap"]
                )

            if operation == "substitution":
                idx = self._rng.choice(alpha_indices)
                original_char = chars[idx]
                neighbors = self.keyboard_neighbors.get(
                    original_char.lower()
                )
                if not neighbors:
                    continue
                new_char = self._rng.choice(neighbors)
                # Preserve upper/lower case of the original character.
                chars[idx] = (
                    new_char.upper()
                    if original_char.isupper()
                    else new_char
                )
                edits_done += 1
                operations_used.append("substitution")

            elif operation == "delete":
                if len(chars) <= 1:
                    continue
                idx = self._rng.choice(alpha_indices)
                del chars[idx]
                edits_done += 1
                operations_used.append("delete")

            elif operation == "insert":
                idx = self._rng.choice(alpha_indices)
                original_char = chars[idx]
                neighbors = self.keyboard_neighbors.get(
                    original_char.lower()
                )
                if not neighbors:
                    continue
                new_char = self._rng.choice(neighbors)
                chars.insert(idx + 1, new_char)
                edits_done += 1
                operations_used.append("insert")

            elif operation == "swap":
                # Find a position where two adjacent characters are both
                # letters, then swap them.
                swappable = [
                    i
                    for i in range(len(chars) - 1)
                    if chars[i].isalpha() and chars[i + 1].isalpha()
                ]
                if not swappable:
                    continue
                idx = self._rng.choice(swappable)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                edits_done += 1
                operations_used.append("swap")

        return "".join(chars), operations_used


# =============================================================================
# PERTURBATION ENGINE
# =============================================================================

class PerturbationEngine:
    """
    Main perturbation engine.

    Two independent perturbation methods live here:

      1. Semantic word substitution (levels: low / medium / high)
         Swaps whole words for thesaurus synonyms, filtered by IndoBERT
         contextual similarity. Intensity = fraction of WORDS changed
         (10% / 20% / 30%).

      2. Character typo perturbation (levels: typo_low / typo_medium / typo_high)
         Introduces keyboard-style character typos. Intensity = fraction of
         CHARACTERS edited (10% / 30% / 50%).

    The two methods use DISTINCT level names, so a given level always maps to
    exactly one method and one intensity. There is no overloading.
    """

    def __init__(
        self,
        thesaurus_path: str,
        random_seed: int = 42,
        sim_min: float = SIM_MIN,
        sim_max: float = SIM_MAX,
    ):

        self.random_seed = random_seed

        self.sim_min = sim_min
        self.sim_max = sim_max

        self.perturbation = (
            SemanticWordSubstitution(
                thesaurus_path=thesaurus_path,
                random_seed=random_seed,
                sim_min=sim_min,
                sim_max=sim_max,
            )
        )

        # Character typo perturbation (levels: typo_low / typo_medium / typo_high).
        # It is fully independent of the semantic word substitution above.
        self.typo_perturbation = TypoCharacterPerturbation(
            random_seed=random_seed,
        )

        logger.info(
            "PerturbationEngine initialized"
        )

        logger.info(
            "Method 1: semantic similarity-based "
            "word substitution"
        )

        logger.info(
            "Low=%.0f%%, Medium=%.0f%%, High=%.0f%%",
            PERTURBATION_INTENSITIES["low"] * 100,
            PERTURBATION_INTENSITIES["medium"] * 100,
            PERTURBATION_INTENSITIES["high"] * 100,
        )

        logger.info(
            "Word cosine similarity range: %.2f-%.2f",
            sim_min,
            sim_max,
        )

        logger.info(
            "Method 2: character typo perturbation"
        )

        logger.info(
            "typo_low=%.0f%%, typo_medium=%.0f%%, typo_high=%.0f%% of characters",
            TYPO_INTENSITIES["typo_low"] * 100,
            TYPO_INTENSITIES["typo_medium"] * 100,
            TYPO_INTENSITIES["typo_high"] * 100,
        )

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def apply_perturbation(
        self,
        text: str,
        level: str,
        intensity: Optional[float] = None,
    ) -> str:

        result = self.apply_perturbation_with_metadata(
            text=text,
            level=level,
            intensity=intensity,
        )

        return result["perturbed_text"]

    def apply_perturbation_with_metadata(
        self,
        text: str,
        level: str,
        intensity: Optional[float] = None,
    ) -> Dict[str, object]:

        level = level.lower().strip()

        # -----------------------------------------------------------------
        # Route to the correct method based on the level name.
        #   low / medium / high            -> semantic word substitution
        #   typo_low / typo_medium / typo_high -> character typo perturbation
        # -----------------------------------------------------------------
        if level in PERTURBATION_INTENSITIES:
            if intensity is None:
                intensity = PERTURBATION_INTENSITIES[level]

            result = self.perturbation.perturb_with_metadata(
                text=text,
                intensity=intensity,
            )

        elif level in TYPO_INTENSITIES:
            if intensity is None:
                intensity = TYPO_INTENSITIES[level]

            result = self.typo_perturbation.perturb_with_metadata(
                text=text,
                intensity=intensity,
            )

        else:
            raise ValueError(
                f"Invalid perturbation level: {level}. Choose one of: "
                f"{list(PERTURBATION_INTENSITIES) + list(TYPO_INTENSITIES)}."
            )

        result["perturbation_level"] = level

        return result

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        level: str = "low",
        intensity: Optional[float] = None,
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        level = level.lower().strip()

        # Accept both the semantic-substitution levels and the typo levels.
        if (
            level not in PERTURBATION_INTENSITIES
            and level not in TYPO_INTENSITIES
        ):
            raise ValueError(
                f"Invalid level: {level}. Choose one of: "
                f"{list(PERTURBATION_INTENSITIES) + list(TYPO_INTENSITIES)}."
            )

        logger.info(
            "Applying %s perturbation to %d rows",
            level.upper(),
            len(df),
        )

        results = []

        for idx, value in tqdm(
            df[text_column].items(),
            total=len(df),
            desc=f"Perturbing {level.upper()}",
        ):

            text = (
                ""
                if pd.isna(value)
                else str(value)
            )

            result = (
                self.apply_perturbation_with_metadata(
                    text=text,
                    level=level,
                    intensity=intensity,
                )
            )

            results.append(
                {
                    "original_index": idx,
                    **result,
                }
            )

        result_df = pd.DataFrame(results)

        # -----------------------------------------------------------------
        # Define what counts as a SUCCESSFUL perturbation.
        #
        # The two methods have different success criteria, so we branch
        # explicitly rather than hiding the difference in one expression:
        #
        #   Semantic word substitution:
        #       changed AND word-similarity in [0.80, 0.95] AND hit word target
        #       (here cosine similarity IS a gate).
        #
        #   Typo perturbation:
        #       changed AND actual character edit ratio hit its target band.
        #       Cosine similarity is recorded as a diagnostic only, so it is
        #       intentionally NOT part of the success test.
        # -----------------------------------------------------------------
        if level in TYPO_INTENSITIES:
            result_df["perturbation_success"] = (
                (~result_df["is_same_as_original"])
                & result_df["perturbation_in_range"]
            )
        else:
            result_df["perturbation_success"] = (
                (~result_df["is_same_as_original"])
                & result_df["similarity_in_range"]
                & result_df["perturbation_in_range"]
            )

        successful_count = int(
            result_df["perturbation_success"].sum()
        )

        perturbed_df = df.copy()

        # Keep original dataframe columns intact
        perturbed_df[text_column] = (
            result_df["perturbed_text"].values
        )

        # Add audit metadata
        metadata_columns = [
            "original_text",
            "perturbed_text",
            "perturbation_level",
            "target_intensity",

            "failed_lookup_words",

            "total_words",
            "eligible_words",
            "target_words",
            "words_changed",

            "actual_ratio_all_words",

            "similarity_score_mean",
            "similarity_score_min",
            "similarity_score_max",

            "is_same_as_original",
            "similarity_in_range",
            "perturbation_in_range",
            "perturbation_success",

            "diagnostic_stats",
            "word_diagnostics",

            "replacements",

            # Typo-specific columns (only present on typo_* runs; guarded below
            # so semantic runs simply skip them).
            "char_edit_ratio",
            "char_cosine_similarity",
            "total_chars",
            "typo_operations",
        ]

        for column in metadata_columns:
            if column in result_df.columns:
                perturbed_df[column] = (
                    result_df[column].values
                )


        # -------------------------------------------------------------------------
        # SUCCESS COUNT
        # -------------------------------------------------------------------------

        successful = (
            result_df["perturbation_in_range"]
            & ~result_df["is_same_as_original"]
        )

        successful_count = int(successful.sum())
        total_count = len(result_df)
        success_percentage = (
            successful_count / total_count * 100
            if total_count
            else 0.0
        )

        # -------------------------------------------------------------------------
        # DATASET-LEVEL PERTURBATION STATISTICS
        # -------------------------------------------------------------------------

        perturbation_stats = {
            "mean_word_change_rate": float(
                result_df["actual_ratio_all_words"].mean()
            ) if total_count else 0.0,

            "mean_similarity_score": float(
                result_df["similarity_score_mean"].dropna().mean()
            )
            if result_df["similarity_score_mean"].notna().any()
            else None,

            "min_similarity_score": float(
                result_df["similarity_score_min"].dropna().min()
            )
            if result_df["similarity_score_min"].notna().any()
            else None,

            "max_similarity_score": float(
                result_df["similarity_score_max"].dropna().max()
            )
            if result_df["similarity_score_max"].notna().any()
            else None,
        }

        perturbed_df.attrs["perturbation_stats"] = perturbation_stats

        # -------------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------------

        logger.info(
            "%s successful perturbations: %d/%d (%.2f%%)",
            level.upper(),
            successful_count,
            total_count,
            success_percentage,
        )

        print(
            f"\n{level.upper()} SUCCESS: "
            f"{successful_count}/{total_count} "
            f"({success_percentage:.2f}%)"
        )

        # -------------------------------------------------------------------------
        # Statistics
        # -------------------------------------------------------------------------

        logger.info(
            "%s target intensity: %.2f%%",
            level.upper(),
            (
                result_df["target_intensity"].iloc[0] * 100
                if len(result_df)
                else 0.0
            ),
        )

        logger.info(
            "%s actual ratio over all words: %.4f",
            level.upper(),
            result_df[
                "actual_ratio_all_words"
            ].mean(),
        )

        logger.info(
            "%s mean similarity score: %.4f",
            level.upper(),
            result_df[
                "similarity_score_mean"
            ].mean(),
        )

        # -------------------------------------------------------------------------
        # Debug samples
        # -------------------------------------------------------------------------

        dbg_perturbation_samples(
            level=level,
            domain="dataframe",
            originals=result_df[
                "original_text"
            ].tolist(),
            perturbed=result_df[
                "perturbed_text"
            ].tolist(),
        )

        dbg_perturbation_stats(
            level=level,
            domain="dataframe",
            n_texts=len(perturbed_df),
            char_change_mean=0.0,
            word_change_mean=float(
                result_df[
                    "actual_ratio_all_words"
                ].mean()
            ),
        )

        # -------------------------------------------------------------------------
        # SAVE
        # -------------------------------------------------------------------------

        if output_csv:

            output_dir = os.path.dirname(
                os.path.abspath(output_csv)
            )

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            perturbed_df.to_csv(
                output_csv,
                index=False,
                encoding="utf-8-sig",
            )

            logger.info(
                "Saved perturbation dataset: %s",
                output_csv,
            )

            print(
                f"Saved → {output_csv}"
            )

        return perturbed_df

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_perturbation_stats(
        self,
        original: str,
        perturbed: str,
    ) -> Dict[str, float]:

        original_words = tokenize_words(
            original
        )

        perturbed_words = tokenize_words(
            perturbed
        )

        if len(original_words) != len(
            perturbed_words
        ):
            raise ValueError(
                "Word substitution should preserve "
                "the number of word tokens."
            )

        words_changed = sum(
            a.lower() != b.lower()
            for a, b in zip(
                original_words,
                perturbed_words,
            )
        )

        ratio = (
            words_changed
            / len(original_words)
            if original_words
            else 0.0
        )

        return {
            "original_word_count": len(
                original_words
            ),
            "perturbed_word_count": len(
                perturbed_words
            ),
            "words_changed": words_changed,
            "word_change_ratio": ratio,
            "is_same_as_original": (
                original == perturbed
            ),
        }

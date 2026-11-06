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
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

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

        # The REQUESTED target is the number of word changes the intensity asks
        # for (10% / 20% / 30% of all words). We never shrink it: success is
        # always judged against this original request.
        requested_target_count = int(
            np.floor(
                len(words) * intensity + 0.5
            )
        )

        # We may not HAVE enough eligible words to reach the requested target.
        # Record that fact; it marks the row as a failed perturbation later,
        # even though we still change as many words as we can.
        insufficient_eligible_words = (
            len(eligible) < requested_target_count
        )

        # The number of words we will actually try to change is capped by how
        # many eligible words exist. This is ONLY used for selection, never for
        # judging success.
        actual_target_count = min(
            requested_target_count,
            len(eligible),
        )

        if requested_target_count == 0:
            return self._infeasible_result(
                text,
                "TARGET_COUNT_ROUNDED_TO_ZERO",
                target_intensity=intensity,
                total_words=len(words),
                eligible_words=len(eligible),
                target_words=requested_target_count,
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
            if len(replacements) >= actual_target_count:
                break

            candidates = item[
                "candidates"
            ]

            # Pick the candidate with the highest contextual cosine similarity.
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
        # Cannot change ANY word -> truly infeasible (headline would be
        # unchanged, so there is nothing partial to keep).
        #
        # NOTE: we deliberately do NOT bail out just because we changed fewer
        # words than the REQUESTED target. When there are too few eligible
        # words, we keep the partially-perturbed headline for analysis and let
        # the success flags below mark it as a FAILED perturbation
        # (insufficient_eligible_words = True).
        # ---------------------------------------------------------------------

        if len(replacements) == 0:
            return self._infeasible_result(
                text,
                "INSUFFICIENT_VALID_REPLACEMENTS",
                target_intensity=intensity,
                total_words=len(words),
                eligible_words=len(eligible),
                target_words=requested_target_count,
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

        # ---------------------------------------------------------------------
        # Validate every replacement
        # ---------------------------------------------------------------------
        logger.info(
            "DEBUG replacements=%d requested_target=%d actual_target=%d eligible=%d",
            len(replacements),
            requested_target_count,
            actual_target_count,
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

        is_same_as_original = (text == perturbed_text)

        # Did we change EXACTLY the requested number of words? This is judged
        # against the ORIGINAL requested target, never the eligibility-capped
        # one, so a partial perturbation (too few eligible words) does not count
        # as hitting the target.
        perturbation_in_range = (
            words_changed == requested_target_count
        )

        # A perturbation SUCCEEDS only when all gates hold:
        #   1. the headline actually changed,
        #   2. every replacement's cosine is in [sim_min, sim_max] (0.80-0.95),
        #   3. it hit the requested token target (10% / 20% / 30%), and
        #   4. there were enough eligible words to meet that request.
        # (3) and (4) overlap in most cases, but we keep both so the reason is
        # explicit and auditable.
        perturbation_success = (
            (not is_same_as_original)
            and similarity_valid
            and perturbation_in_range
            and (not insufficient_eligible_words)
        )

        return {
            "original_text": text,
            "perturbed_text": perturbed_text,

            "perturbation_level": None,

            "target_intensity": intensity,

            "total_words": len(words),

            "failed_lookup_words": lookup_stats["failed"],

            "eligible_words": len(eligible),
            # The requested target (never capped) is what success is judged on.
            "target_words": requested_target_count,
            # How many we could actually attempt given the eligible words.
            "actual_target_words": actual_target_count,

            "can_satisfy_target": (
                not insufficient_eligible_words
            ),
            "eligible_shortage": insufficient_eligible_words,
            "insufficient_eligible_words": insufficient_eligible_words,
            "feasibility_reason": (
                "INSUFFICIENT_ELIGIBLE_WORDS"
                if insufficient_eligible_words
                else "SUFFICIENT_ELIGIBLE_WORDS"
            ),

            "words_changed": words_changed,

            "actual_ratio_all_words": actual_ratio_all_words,

            "diagnostic_stats": diagnostic_stats,
            "word_diagnostics": word_diagnostics,

            "is_same_as_original": is_same_as_original,

            "similarity_in_range": similarity_valid,

            "perturbation_in_range": perturbation_in_range,

            # Final success flag: all four gates above. Kept here so every
            # consumer (evaluation engine, pattern analysis) sees one identical
            # definition.
            "perturbation_success": perturbation_success,

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

            "actual_target_words": valid_replacements,

            "can_satisfy_target": (
                eligible_words >= target_words
            ),

            "eligible_shortage": (
                eligible_words < target_words
            ),

            "insufficient_eligible_words": (
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

            # Infeasible attempts are always failed perturbations.
            "perturbation_success": False,

            "similarity_score_mean": np.nan,

            "similarity_score_min": np.nan,

            "similarity_score_max": np.nan,

            "replacements": [],
        }


# =============================================================================
# PERTURBATION ENGINE
# =============================================================================

class PerturbationEngine:
    """
    Main perturbation engine.

    One perturbation method lives here:

      Semantic word substitution (levels: low / medium / high)
      Swaps whole words for thesaurus synonyms, filtered by IndoBERT
      contextual similarity. Intensity = fraction of WORDS changed
      (10% / 20% / 30%).
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

        logger.info(
            "PerturbationEngine initialized"
        )

        logger.info(
            "Method: semantic similarity-based "
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
        # Only the semantic word-substitution levels are supported.
        #   low / medium / high -> semantic word substitution
        # -----------------------------------------------------------------
        if level in PERTURBATION_INTENSITIES:
            if intensity is None:
                intensity = PERTURBATION_INTENSITIES[level]

            result = self.perturbation.perturb_with_metadata(
                text=text,
                intensity=intensity,
            )

        else:
            raise ValueError(
                f"Invalid perturbation level: {level}. Choose one of: "
                f"{list(PERTURBATION_INTENSITIES)}."
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

        # Accept only the semantic word-substitution levels.
        if level not in PERTURBATION_INTENSITIES:
            raise ValueError(
                f"Invalid level: {level}. Choose one of: "
                f"{list(PERTURBATION_INTENSITIES)}."
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
        # Semantic word substitution succeeds only when ALL FOUR gates hold:
        #     1. the headline actually changed,
        #     2. word-similarity in [0.80, 0.95] (cosine IS a gate),
        #     3. it hit the requested word target (10% / 20% / 30%), and
        #     4. there were enough eligible words to meet that request.
        #
        # Gate 4 is what keeps a PARTIAL perturbation (too few eligible words)
        # out of the success set, even though we still saved its changed text.
        # -----------------------------------------------------------------
        result_df["perturbation_success"] = (
            (~result_df["is_same_as_original"])
            & result_df["similarity_in_range"]
            & result_df["perturbation_in_range"]
            & (~result_df["insufficient_eligible_words"])
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

            "actual_target_words",

            "is_same_as_original",
            "similarity_in_range",
            "perturbation_in_range",
            "insufficient_eligible_words",
            "eligible_shortage",
            "feasibility_reason",
            "perturbation_success",

            "diagnostic_stats",
            "word_diagnostics",

            "replacements",
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

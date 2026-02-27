"""
rag_system.py - BM25-based Retrieval-Augmented Generation for Yale dining menus.

No external vector DB required - implemented from scratch using BM25 (Okapi BM25),
the gold standard for keyword-based information retrieval.

Architecture:
  - Each menu item becomes a searchable "document" with rich text representation
  - BM25 scores documents by term frequency + inverse document frequency
  - Dietary restriction filtering is applied as a hard constraint post-retrieval
  - Agents query this RAG via tool calls to ground their arguments in real data
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any


# BM25 tuning parameters (standard defaults)
BM25_K1 = 1.5   # term frequency saturation
BM25_B  = 0.75  # document length normalization

# Keyword synonyms for dietary restriction matching
RESTRICTION_SYNONYMS = {
    "vegan":       ["vegan", "plant-based"],
    "vegetarian":  ["vegetarian", "vegan"],
    "gluten-free": ["gluten", "gluten-free", "gluten free"],
    "halal":       ["halal"],
    "kosher":      ["kosher"],
    "dairy-free":  ["dairy", "dairy-free", "dairy free", "lactose"],
    "nut-free":    ["nut", "peanut", "tree nut"],
    "low-calorie": [],  # handled separately via nutrition
    "high-protein": [], # handled separately via nutrition
}


class MenuRAGSystem:
    """
    In-memory BM25 retrieval over Yale dining hall menu items.

    Usage:
        rag = MenuRAGSystem()
        rag.add_document(doc_id, text, metadata)
        ...
        rag.build_index()
        results = rag.search("high protein chicken", top_k=10)
    """

    def __init__(self):
        self.documents: List[Tuple[str, str, dict]] = []   # (id, text, metadata)
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self._built = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_document(self, doc_id: str, text: str, metadata: dict) -> None:
        tokens = self._tokenize(text)
        self.documents.append((doc_id, text, metadata))
        self.doc_lengths[doc_id] = len(tokens)

        tf: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        for term, count in tf.items():
            self.inverted_index[term].append((doc_id, count))

    def build_index(self) -> None:
        n = len(self.documents)
        if n == 0:
            return
        self.avg_doc_length = sum(self.doc_lengths.values()) / n
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)
        self._built = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _bm25_score(self, doc_id: str, query_tokens: List[str]) -> float:
        score = 0.0
        dl = self.doc_lengths.get(doc_id, 0)
        norm = 1 - BM25_B + BM25_B * dl / max(self.avg_doc_length, 1)
        for tok in query_tokens:
            idf = self.idf.get(tok, 0.0)
            if idf == 0:
                continue
            for did, tf in self.inverted_index.get(tok, []):
                if did == doc_id:
                    score += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * norm)
                    break
        return score

    def _matches_dietary(self, metadata: dict, restrictions: List[str]) -> bool:
        """
        Hard filter: all specified restrictions must be satisfied.
        A restriction is satisfied if any dietary_flag on the item contains the keyword.
        Special cases: 'high-protein' and 'low-calorie' use numeric thresholds.
        """
        flags_lower = [f.lower() for f in metadata.get("dietary_flags", [])]
        flags_str = " ".join(flags_lower)

        for restriction in restrictions:
            restriction = restriction.lower().strip()

            # Numeric shortcuts
            if restriction in ("high-protein", "high protein"):
                protein = metadata.get("protein_g") or 0
                try:
                    if float(protein) < 15:
                        return False
                except (TypeError, ValueError):
                    return False
                continue
            if restriction in ("low-calorie", "low calorie"):
                cal = metadata.get("calories") or 9999
                try:
                    if float(cal) > 400:
                        return False
                except (TypeError, ValueError):
                    return False
                continue

            # Keyword matching against dietary flags
            synonyms = RESTRICTION_SYNONYMS.get(restriction, [restriction])
            if not any(syn in flags_str for syn in synonyms):
                return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        hall_filter: Optional[str] = None,
        dietary_filter: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, dict, float]]:
        """
        Returns list of (doc_id, text, metadata, score) sorted by relevance.
        dietary_filter items are applied as hard constraints BEFORE scoring.
        """
        if not self._built:
            self.build_index()

        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}

        for doc_id, text, metadata in self.documents:
            # Hard filters first (cheap)
            if hall_filter and metadata.get("hall", "").lower() != hall_filter.lower():
                continue
            if dietary_filter and not self._matches_dietary(metadata, dietary_filter):
                continue
            scores[doc_id] = self._bm25_score(doc_id, query_tokens)

        # Build result list sorted by score
        doc_map = {d[0]: d for d in self.documents}
        results = [
            (doc_id, doc_map[doc_id][1], doc_map[doc_id][2], score)
            for doc_id, score in scores.items()
            if score > 0
        ]
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def get_hall_items(self, hall_name: str) -> List[dict]:
        """Return all metadata dicts for a given dining hall."""
        return [
            meta for _, _, meta in self.documents
            if meta.get("hall", "").lower() == hall_name.lower()
        ]

    def get_all_halls(self) -> List[str]:
        seen = []
        for _, _, meta in self.documents:
            h = meta.get("hall", "")
            if h and h not in seen:
                seen.append(h)
        return seen

    @property
    def total_documents(self) -> int:
        return len(self.documents)


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------

def build_rag_from_menus(all_menus: Dict[str, Dict[str, List[dict]]]) -> MenuRAGSystem:
    """
    Convert fetched dining hall menus into a searchable BM25 index.
    Each food item becomes a rich-text document.
    """
    rag = MenuRAGSystem()

    for hall_name, menu in all_menus.items():
        for station, items in menu.items():
            for item in items:
                flags_str = (
                    ", ".join(item["dietary_flags"]) if item["dietary_flags"] else "no special flags"
                )
                cal_str    = f"{int(item['calories'])} calories" if item.get("calories") else ""
                protein_str = f"{item['protein_g']} grams protein" if item.get("protein_g") else ""
                carbs_str   = f"{item.get('carbs_g')} grams carbs" if item.get("carbs_g") else ""
                fat_str     = f"{item.get('fat_g')} grams fat" if item.get("fat_g") else ""
                fiber_str   = f"{item.get('fiber_g')} grams fiber" if item.get("fiber_g") else ""

                # Nutrient-level descriptors (boost searchability)
                nutrient_tags = []
                try:
                    if item.get("protein_g") and float(item["protein_g"]) >= 20:
                        nutrient_tags.append("high protein")
                    if item.get("calories") and float(item["calories"]) <= 300:
                        nutrient_tags.append("low calorie light")
                    if item.get("calories") and float(item["calories"]) >= 700:
                        nutrient_tags.append("hearty filling")
                    if item.get("fiber_g") and float(item["fiber_g"]) >= 5:
                        nutrient_tags.append("high fiber")
                except (TypeError, ValueError):
                    pass

                # Full-text representation for BM25
                text = (
                    f"{item['name']} served at {hall_name} dining hall "
                    f"in the {station} station. "
                    f"Dietary flags: {flags_str}. "
                    + (f"{cal_str}. " if cal_str else "")
                    + (f"{protein_str}. " if protein_str else "")
                    + (f"{carbs_str}. " if carbs_str else "")
                    + (f"{fat_str}. " if fat_str else "")
                    + (f"{fiber_str}. " if fiber_str else "")
                    + (" ".join(nutrient_tags) + ". " if nutrient_tags else "")
                    + (item.get("description", "") or "")
                )

                doc_id = f"{hall_name}::{station}::{item['name']}"
                metadata = {
                    **item,
                    "hall":    hall_name,
                    "station": station,
                }
                rag.add_document(doc_id, text, metadata)

    rag.build_index()
    return rag


def score_halls_for_user(
    prefs: dict,
    rag: MenuRAGSystem,
    top_n: int = 4,
) -> List[Tuple[str, float]]:
    """
    Aggregate BM25 scores per dining hall based on user preferences.
    Returns [(hall_name, score), ...] sorted by relevance.
    """
    # Build a unified preference query
    query_parts = [
        prefs.get("goal", ""),
        prefs.get("preferences", ""),
        prefs.get("restrictions", ""),
    ]
    query = " ".join(p for p in query_parts if p)

    # Determine dietary hard filters (only well-known restrictions)
    dietary_filter = []
    restrictions_raw = prefs.get("restrictions", "").lower()
    for key in RESTRICTION_SYNONYMS:
        if key.replace("-", " ") in restrictions_raw or key in restrictions_raw:
            dietary_filter.append(key)

    results = rag.search(query, top_k=200, dietary_filter=dietary_filter or None)

    hall_scores: Dict[str, float] = defaultdict(float)
    for _, _, meta, score in results:
        hall_scores[meta.get("hall", "")] += score

    ranked = sorted(hall_scores.items(), key=lambda x: x[1], reverse=True)
    top = [h for h, _ in ranked[:top_n] if h]

    # Ensure minimum 3 contenders
    all_halls = rag.get_all_halls()
    for h in all_halls:
        if len(top) >= top_n:
            break
        if h not in top:
            top.append(h)

    return [(h, hall_scores.get(h, 0.0)) for h in top]

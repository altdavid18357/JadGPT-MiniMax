#!/usr/bin/env python3
"""
run_debate.py — JadGPT Dining Hall Meal Recommender CLI

Usage:
    python run_debate.py                  # interactive mode
    python run_debate.py --meal lunch     # force a specific meal
    python run_debate.py --demo           # run with preset demo preferences
    python run_debate.py --demo-vegan     # run with preset vegan preferences

Architecture:
    1. Fetch live menu from Yale dining halls (parallel HTTP)
    2. Build BM25 RAG index over all menu items
    3. Single agent searches the menu and recommends the best meal
       options tailored to the user's goals and restrictions
"""

import sys
import argparse
from datetime import date

try:
    from menu_fetcher import fetch_all_menus
except ImportError as e:
    print(f"[ERROR] Cannot import menu_fetcher: {e}")
    sys.exit(1)

try:
    from rag_system import build_rag_from_menus
except ImportError as e:
    print(f"[ERROR] Cannot import rag_system: {e}")
    sys.exit(1)

try:
    from debate_agents import run_recommender
except ImportError as e:
    print(f"[ERROR] Cannot import debate_agents: {e}")
    sys.exit(1)


BANNER = r"""
     _           _  ____ ____ _____
    | | __ _  __| |/ ___|  _ \_   _|
 _  | |/ _` |/ _` | |  _| |_) || |
| |_| | (_| | (_| | |_| |  __/ | |
 \___/ \__,_|\__,_|\____|_|    |_|

  Yale Dining Hall Meal Recommender
  Powered by MiniMax AI (via Anthropic SDK)
  ─────────────────────────────────────────
"""

DEMO_PREFS = {
    "goal":         "high protein muscle building",
    "restrictions": "none",
    "allergies":    "none",
    "preferences":  "savory, meat-forward, hearty",
}

DEMO_PREFS_VEGAN = {
    "goal":         "balanced vegan meal with enough protein",
    "restrictions": "vegan",
    "allergies":    "none",
    "preferences":  "fresh vegetables, whole grains, flavorful",
}


def _prompt(label: str, hint: str = "") -> str:
    full = f"  {label}"
    if hint:
        full += f" ({hint})"
    full += ": "
    val = input(full).strip()
    return val or "none"


def gather_preferences() -> dict:
    print("\n  Tell us about your dietary needs and goals:\n")
    return {
        "goal":         _prompt("Primary goal",    "e.g. high protein, weight loss, comfort food, balanced meal"),
        "restrictions": _prompt("Restrictions",    "e.g. vegan, vegetarian, gluten-free, halal, none"),
        "allergies":    _prompt("Allergies",       "e.g. nuts, dairy, shellfish, none"),
        "preferences":  _prompt("Flavor/cuisine",  "e.g. spicy, Asian, Italian, savory, none"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JadGPT — Yale Dining Hall Meal Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--meal",
        choices=["breakfast", "lunch", "dinner"],
        help="Override current meal detection",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with preset demo preferences (high protein)",
    )
    parser.add_argument(
        "--demo-vegan",
        action="store_true",
        help="Run with preset vegan demo preferences",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(BANNER)

    # ── Preferences ──────────────────────────────────────────────
    if args.demo:
        prefs = DEMO_PREFS
        print("  [DEMO MODE] Using preset preferences:")
        for k, v in prefs.items():
            print(f"    {k}: {v}")
    elif args.demo_vegan:
        prefs = DEMO_PREFS_VEGAN
        print("  [DEMO VEGAN MODE] Using preset vegan preferences:")
        for k, v in prefs.items():
            print(f"    {k}: {v}")
    else:
        prefs = gather_preferences()

    # ── Menu fetching ─────────────────────────────────────────────
    print()
    meal, all_menus = fetch_all_menus(verbose=True)

    if args.meal:
        meal = args.meal
        print(f"  [Override] Using meal: {meal}")

    if not all_menus:
        print("\n[ERROR] No menu data could be fetched. Check your internet connection.")
        sys.exit(1)

    total_items = sum(
        len(items)
        for menu in all_menus.values()
        for items in menu.values()
    )
    print(f"\n  Total items indexed: {total_items} across {len(all_menus)} dining halls")

    # ── RAG index ────────────────────────────────────────────────
    print(f"\n  Building BM25 RAG index... ", end="", flush=True)
    rag = build_rag_from_menus(all_menus)
    print(f"done ({rag.total_documents} documents)")

    # ── Get recommendation ───────────────────────────────────────
    print(f"\n  Meal: {meal.capitalize()} | Date: {date.today().strftime('%B %d, %Y')}")
    print(f"\n{'=' * 62}")
    print(f"  MEAL RECOMMENDATION")
    print(f"{'=' * 62}\n")

    result = run_recommender(prefs, rag, all_menus, meal)
    print(result)

    print(f"\n{'=' * 62}\n")


if __name__ == "__main__":
    main()

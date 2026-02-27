#!/usr/bin/env python3
"""
main.py — JadGPT Flask API server

Endpoints:
  GET/POST /recommend    — filtered meal recommendations from Nutrislice
  GET      /menu/<slug>  — full menu for a specific dining hall
  GET      /halls        — list of all available dining halls
  POST     /agent/recommend — AI-powered meal plan (MiniMax via debate_agents + RAG)
"""

import sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import date
from menu_fetcher import fetch_all_menus, get_hall_list, get_current_meal
from debate_agents import run_recommender
from rag_system import build_rag_from_menus

app = Flask(__name__)
CORS(app)

# ── Simple in-memory cache (10-minute TTL) ───────────────────
_cache: dict = {}

def _get_menus():
    now = time.time()
    if _cache.get("ts") and now - _cache["ts"] < 600:
        return _cache["meal"], _cache["menus"]
    meal, menus = fetch_all_menus(verbose=False)
    _cache.update({"meal": meal, "menus": menus, "ts": now})
    return meal, menus


# ── /recommend ───────────────────────────────────────────────
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    """
    Accept dietary restrictions + calorie/protein goals and return
    a filtered, ranked list of today's menu items across all dining halls.
    """
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
    else:
        body = request.args.to_dict()

    # Parse restrictions (list or comma-separated string)
    restrictions = body.get("restrictions", [])
    if isinstance(restrictions, str):
        restrictions = [r.strip().lower() for r in restrictions.split(",") if r.strip()]
    else:
        restrictions = [r.strip().lower() for r in restrictions if r]

    calorie_goal = int(body.get("calorie_goal", 2000))
    protein_goal = int(body.get("protein_goal", 50))

    meal, all_menus = _get_menus()

    results = []
    for hall_name, menu in all_menus.items():
        for station, items in menu.items():
            for item in items:
                # Dietary restriction filtering (case-insensitive substring match)
                if restrictions:
                    flags_lower = " ".join(f.lower() for f in item.get("dietary_flags", []))
                    if not all(r in flags_lower for r in restrictions):
                        continue

                results.append({
                    "name":          item["name"],
                    "calories":      item.get("calories"),
                    "protein_g":     item.get("protein_g"),
                    "carbs_g":       item.get("carbs_g"),
                    "fat_g":         item.get("fat_g"),
                    "dietary_flags": item.get("dietary_flags", []),
                    "dining_hall":   hall_name,
                    "station":       station,
                })

    # Rank by protein content (descending); items with no data sink to bottom
    results.sort(key=lambda x: (x.get("protein_g") or 0), reverse=True)

    # Pick the best unique-named item per dining hall, then return top 3 halls
    hall_top: dict = {}
    seen_names: set = set()
    for item in results:
        hall = item["dining_hall"]
        key = item["name"].lower().strip()
        if hall not in hall_top and key not in seen_names:
            hall_top[hall] = item
            seen_names.add(key)

    diverse_results = sorted(
        hall_top.values(),
        key=lambda x: x.get("protein_g") or 0,
        reverse=True,
    )[:3]

    return jsonify({
        "meal":            meal,
        "date":            date.today().isoformat(),
        "calorie_goal":    calorie_goal,
        "protein_goal":    protein_goal,
        "total_found":     len(results),
        "recommendations": diverse_results,
    })


# ── /menu/<hall_slug> ────────────────────────────────────────
@app.route("/menu/<hall_slug>", methods=["GET"])
def get_hall_menu(hall_slug):
    """Return today's full menu for a specific dining hall by Nutrislice slug."""
    meal, all_menus = _get_menus()

    # Try fuzzy name match first
    matched_name = None
    for name in all_menus:
        if hall_slug.replace("-", " ").lower() in name.lower():
            matched_name = name
            break

    if not matched_name or matched_name not in all_menus:
        return jsonify({
            "error":     f'No menu data for "{hall_slug}"',
            "available": list(all_menus.keys()),
        }), 404

    return jsonify({
        "hall": matched_name,
        "meal": meal,
        "date": date.today().isoformat(),
        "menu": all_menus[matched_name],
    })


# ── /halls ───────────────────────────────────────────────────
@app.route("/halls", methods=["GET"])
def get_halls():
    """Return all dining hall names and their Nutrislice slugs."""
    return jsonify({"halls": get_hall_list()})


# ── Helpers ──────────────────────────────────────────────────
def _extract_picks(meal_plan: str, all_menus: dict) -> list:
    """
    Parse structured item picks from the AI meal plan text.
    Looks for lines like:
      1. Dish Name — portion
         📍 Hall Name · Station
    and resolves them against the live menu data.
    """
    picks = []
    # Split on numbered items (1. / 2. / 3.)
    for block in re.split(r'\n(?=\d+\.)', meal_plan):
        name_m = re.match(r'\d+\.\s+(.+?)\s+—', block)
        hall_m = re.search(r'📍\s+(.+?)\s+[·•]', block)
        if not name_m or not hall_m:
            continue
        dish_name = name_m.group(1).strip().lower()
        hall_hint = hall_m.group(1).strip().lower()

        for hall, menu in all_menus.items():
            if hall_hint not in hall.lower() and hall.lower() not in hall_hint:
                continue
            for station, items in menu.items():
                for item in items:
                    if item["name"].lower() == dish_name:
                        picks.append({
                            **item,
                            "dining_hall": hall,
                            "station":     station,
                        })
                        break
                else:
                    continue
                break
            else:
                continue
            break

    return picks


# ── /agent/recommend ─────────────────────────────────────────
@app.route("/agent/recommend", methods=["POST"])
def agent_recommend():
    """
    AI-powered meal plan: uses MiniMax agent to build a portion-aware
    combination of dishes that together hit the user's calorie/protein goals.
    """
    body = request.get_json(silent=True) or {}

    restrictions = body.get("restrictions", [])
    if isinstance(restrictions, str):
        restrictions = [r.strip() for r in restrictions.split(",") if r.strip()]

    prefs = {
        "calorie_goal":  int(body.get("calorie_goal", 2000)),
        "protein_goal":  int(body.get("protein_goal", 50)),
        "restrictions":  ", ".join(restrictions) if restrictions else "none",
        "allergies":     body.get("allergies", "none"),
        "preferences":   body.get("preferences", "no specific preference"),
    }

    try:
        meal, all_menus = _get_menus()
        rag = build_rag_from_menus(all_menus)
        meal_plan = run_recommender(prefs, rag, all_menus, meal)
        picks     = _extract_picks(meal_plan, all_menus)
        return jsonify({
            "meal":      meal,
            "date":      date.today().isoformat(),
            "meal_plan": meal_plan,
            "picks":     picks,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /health ──────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "meal": get_current_meal()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

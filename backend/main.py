#!/usr/bin/env python3
"""
main.py — JadGPT Flask API server

Endpoints:
  GET/POST /recommend    — filtered meal recommendations from Nutrislice
  GET      /menu/<slug>  — full menu for a specific dining hall
  GET      /halls        — list of all available dining halls
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import date
from menu_fetcher import fetch_all_menus, YALE_DINING_HALLS, get_current_meal

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

    return jsonify({
        "meal":            meal,
        "date":            date.today().isoformat(),
        "calorie_goal":    calorie_goal,
        "protein_goal":    protein_goal,
        "total_found":     len(results),
        "recommendations": results[:24],
    })


# ── /menu/<hall_slug> ────────────────────────────────────────
@app.route("/menu/<hall_slug>", methods=["GET"])
def get_hall_menu(hall_slug):
    """Return today's full menu for a specific dining hall by Nutrislice slug."""
    meal, all_menus = _get_menus()

    # Direct slug match against YALE_DINING_HALLS
    matched_name = None
    for name, slug in YALE_DINING_HALLS.items():
        if slug == hall_slug or slug.rstrip("-college") == hall_slug.rstrip("-college"):
            matched_name = name
            break

    # Fallback: fuzzy name match
    if not matched_name:
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
    return jsonify({
        "halls": [
            {"name": name, "slug": slug}
            for name, slug in YALE_DINING_HALLS.items()
        ]
    })


# ── /health ──────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "meal": get_current_meal()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

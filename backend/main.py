#!/usr/bin/env python3
"""
main.py — BoolaBites Flask API

Endpoints:
  GET/POST /recommend         — basic filtered meal recommendations (fast fallback)
  GET      /menu/<slug>       — full menu for a specific dining hall
  GET      /halls             — list of all available dining halls
  GET      /health            — health + AI status check
  POST     /agent/recommend   — autonomous AI meal plan (MiniMax via Anthropic SDK)
  POST     /agent/chat        — AI-powered chat response
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import date
from menu_fetcher import fetch_all_menus, YALE_DINING_HALLS, get_current_meal

# ── Optional AI imports ───────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    import anthropic as _anthropic_lib
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════
# MENU CACHE
# ══════════════════════════════════════════════════════════════
_cache: dict = {}

def _get_menus():
    now = time.time()
    if _cache.get("ts") and now - _cache["ts"] < 600:
        return _cache["meal"], _cache["menus"]
    meal, menus = fetch_all_menus(verbose=False)
    _cache.update({"meal": meal, "menus": menus, "ts": now})
    return meal, menus


# ══════════════════════════════════════════════════════════════
# AI CLIENT (MiniMax via Anthropic SDK)
# ══════════════════════════════════════════════════════════════
_ai_client = None
MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5")


def get_ai_client():
    global _ai_client
    if not AI_AVAILABLE:
        return None
    if _ai_client is None:
        api_key  = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        if not api_key:
            return None
        _ai_client = _anthropic_lib.Anthropic(api_key=api_key, base_url=base_url)
    return _ai_client


# ══════════════════════════════════════════════════════════════
# AGENT TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════
AGENT_TOOLS = [
    {
        "name": "get_menu",
        "description": (
            "Fetch today's Yale dining hall menu items from the Nutrislice API. "
            "Returns food items across all dining halls filtered by the user's dietary "
            "restrictions, with nutritional info. Always call this before making recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_type": {
                    "type": "string",
                    "enum": ["breakfast", "lunch", "dinner", "current"],
                    "description": "Which meal to fetch. 'current' auto-detects based on time of day.",
                }
            },
            "required": ["meal_type"],
        },
    },
    {
        "name": "get_nutrition",
        "description": (
            "Look up detailed nutritional info — calories, protein, carbs, fat, fiber — "
            "for specific menu items by name. Use this to compare 3–5 promising candidates "
            "before making your final recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Food item names to look up (partial matches work).",
                }
            },
            "required": ["item_names"],
        },
    },
    {
        "name": "check_user_goals",
        "description": (
            "Retrieve the user's saved dietary restrictions, daily calorie goal, "
            "and protein goal. Always call this first so you can personalise recommendations."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "send_recommendation",
        "description": (
            "Deliver your final personalised meal recommendation. Call this once you have "
            "analysed the menu and identified the best matches for the user's goals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_type": {
                    "type": "string",
                    "description": "The meal being recommended (breakfast / lunch / dinner).",
                },
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":        {"type": "string"},
                            "dining_hall": {"type": "string"},
                            "station":     {"type": "string"},
                            "reason":      {"type": "string",
                                            "description": "Why this item fits the user's specific goals."},
                            "calories":    {"type": "number"},
                            "protein_g":   {"type": "number"},
                        },
                        "required": ["name", "reason"],
                    },
                    "description": "Top 3 recommended items.",
                },
                "summary": {
                    "type": "string",
                    "description": "A concise (≤60 words) natural-language summary of the meal plan.",
                },
            },
            "required": ["meal_type", "recommendations", "summary"],
        },
    },
]


# ══════════════════════════════════════════════════════════════
# TOOL EXECUTION
# ══════════════════════════════════════════════════════════════
def _execute_tool(name: str, tool_input: dict, user_prefs: dict) -> dict:

    # ── check_user_goals ─────────────────────────────────────
    if name == "check_user_goals":
        return {
            "restrictions": user_prefs.get("restrictions", []),
            "calorie_goal": user_prefs.get("calorie_goal", 2000),
            "protein_goal": user_prefs.get("protein_goal", 50),
        }

    # ── get_menu ─────────────────────────────────────────────
    if name == "get_menu":
        meal, all_menus = _get_menus()
        restrictions = [r.lower() for r in user_prefs.get("restrictions", [])]

        items = []
        for hall, menu in all_menus.items():
            for station, station_items in menu.items():
                for item in station_items:
                    if restrictions:
                        flags = " ".join(f.lower() for f in item.get("dietary_flags", []))
                        if not all(r in flags for r in restrictions):
                            continue
                    items.append({
                        "name":          item["name"],
                        "dining_hall":   hall,
                        "station":       station,
                        "calories":      item.get("calories"),
                        "protein_g":     item.get("protein_g"),
                        "dietary_flags": item.get("dietary_flags", []),
                    })

        items.sort(key=lambda x: x.get("protein_g") or 0, reverse=True)
        return {
            "meal":        meal,
            "total_items": len(items),
            "halls":       list(all_menus.keys()),
            "items":       items[:40],   # cap for context window
        }

    # ── get_nutrition ─────────────────────────────────────────
    if name == "get_nutrition":
        _, all_menus = _get_menus()
        targets = [n.lower() for n in tool_input.get("item_names", [])]
        found = []
        for hall, menu in all_menus.items():
            for station, items in menu.items():
                for item in items:
                    if any(t in item["name"].lower() for t in targets):
                        found.append({
                            "name":          item["name"],
                            "dining_hall":   hall,
                            "station":       station,
                            "calories":      item.get("calories"),
                            "protein_g":     item.get("protein_g"),
                            "carbs_g":       item.get("carbs_g"),
                            "fat_g":         item.get("fat_g"),
                            "fiber_g":       item.get("fiber_g"),
                            "dietary_flags": item.get("dietary_flags", []),
                        })
        return {"nutrition_data": found[:15]}

    # ── send_recommendation ───────────────────────────────────
    if name == "send_recommendation":
        return {"status": "delivered", "captured": tool_input}

    return {"error": f"Unknown tool: {name}"}


def _summarize_tool(name: str, result: dict, tool_input: dict) -> str:
    if name == "check_user_goals":
        r = result.get("restrictions") or ["none"]
        return (f"restrictions: {', '.join(r)}  |  "
                f"{result.get('calorie_goal')} kcal goal  |  "
                f"{result.get('protein_goal')}g protein goal")
    if name == "get_menu":
        return (f"{result.get('total_items', 0)} matching items across "
                f"{len(result.get('halls', []))} dining halls")
    if name == "get_nutrition":
        items = result.get("nutrition_data", [])
        if items:
            return "Retrieved: " + ", ".join(i["name"] for i in items[:4])
        return "No matching items found"
    if name == "send_recommendation":
        recs = tool_input.get("recommendations", [])
        return f"{len(recs)} recommendation(s) packaged"
    return ""


# ══════════════════════════════════════════════════════════════
# CORE AGENTIC LOOP
# ══════════════════════════════════════════════════════════════
def run_meal_agent(user_prefs: dict,
                   energy_history: list = None,
                   message: str = None) -> dict:

    client = get_ai_client()
    if not client:
        return {"error": "AI unavailable — set ANTHROPIC_API_KEY in backend/.env",
                "fallback": True}

    meal, _ = _get_menus()

    # ── Build energy context ──────────────────────────────────
    energy_ctx = ""
    if energy_history:
        lines = [
            f"  - {e.get('meal','?')} ({e.get('time','?')}): {e.get('level','?')} energy"
            for e in energy_history[-6:]
        ]
        energy_ctx = "User energy log today:\n" + "\n".join(lines) + "\n\n"
        lows = [e for e in energy_history if e.get("level") == "low"]
        if len(lows) >= 2:
            energy_ctx += (
                "⚠️  User has reported low energy multiple times — "
                "strongly prioritise high-protein, complex-carb options.\n\n"
            )

    system_prompt = f"""You are BoolaBites 🐻, an autonomous Yale dining hall meal advisor.

TOOL WORKFLOW — follow this sequence every time:
1. check_user_goals   → understand the user's dietary profile
2. get_menu           → fetch today's available options (meal_type="current")
3. get_nutrition      → compare 3–5 promising candidates in detail
4. send_recommendation → deliver your final top-3 picks

{energy_ctx}Today: {date.today().strftime('%A, %B %d, %Y')} | Current meal: {meal}

Rules:
- Dietary restrictions are NON-NEGOTIABLE — never recommend a restricted item
- Only recommend items confirmed to exist in the menu data returned by get_menu
- Every recommendation must include calories and protein_g from real data
- The "reason" field must explain specifically how this item meets the user's goals
- summary ≤ 60 words — practical and direct, no filler phrases"""

    user_msg = (message or
                f"It's {meal} time at Yale. Analyse today's menu and give me your top recommendations.")

    messages   = [{"role": "user", "content": user_msg}]
    tool_log   = []
    final_data = None
    final_text = ""

    for _turn in range(10):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=system_prompt,
                tools=AGENT_TOOLS,
                messages=messages,
            )
        except Exception as e:
            return {"error": str(e), "tool_log": tool_log, "fallback": True}

        text_parts, tool_uses = [], []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_uses.append(block)

        final_text = "".join(text_parts)

        if response.stop_reason == "end_turn" or not tool_uses:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            result = _execute_tool(tu.name, tu.input, user_prefs)
            tool_log.append({
                "tool":           tu.name,
                "input":          tu.input,
                "result_summary": _summarize_tool(tu.name, result, tu.input),
            })
            if tu.name == "send_recommendation":
                final_data = tu.input
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tu.id,
                "content":     json.dumps(result)[:3000],
            })

        messages.append({"role": "user", "content": tool_results})

        if final_data:
            # One extra turn to get a closing natural-language sentence
            try:
                closing = client.messages.create(
                    model=MODEL, max_tokens=256,
                    system=system_prompt, tools=AGENT_TOOLS, messages=messages,
                )
                for b in closing.content:
                    if hasattr(b, "text") and b.text.strip():
                        final_text = b.text.strip()
                        break
            except Exception:
                pass
            break

    return {
        "meal":            meal,
        "date":            date.today().isoformat(),
        "tool_log":        tool_log,
        "recommendations": final_data.get("recommendations", []) if final_data else [],
        "summary":         final_data.get("summary", final_text)  if final_data else final_text,
        "final_text":      final_text,
    }


# ══════════════════════════════════════════════════════════════
# AGENT ENDPOINTS
# ══════════════════════════════════════════════════════════════
@app.route("/agent/recommend", methods=["POST"])
def agent_recommend():
    body = request.get_json(silent=True) or {}
    prefs = {
        "restrictions": body.get("restrictions", []),
        "calorie_goal": int(body.get("calorie_goal", 2000)),
        "protein_goal": int(body.get("protein_goal", 50)),
    }
    result = run_meal_agent(prefs, energy_history=body.get("energy_history", []))
    return jsonify(result)


@app.route("/agent/chat", methods=["POST"])
def agent_chat():
    body = request.get_json(silent=True) or {}
    prefs = {
        "restrictions": body.get("restrictions", []),
        "calorie_goal": int(body.get("calorie_goal", 2000)),
        "protein_goal": int(body.get("protein_goal", 50)),
    }
    result = run_meal_agent(
        prefs,
        energy_history=body.get("energy_history", []),
        message=body.get("message", ""),
    )
    return jsonify(result)


# ══════════════════════════════════════════════════════════════
# BASIC (NON-AI) ENDPOINTS — kept as fast fallback
# ══════════════════════════════════════════════════════════════
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
    else:
        body = request.args.to_dict()

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

    results.sort(key=lambda x: (x.get("protein_g") or 0), reverse=True)
    return jsonify({
        "meal":            meal,
        "date":            date.today().isoformat(),
        "calorie_goal":    calorie_goal,
        "protein_goal":    protein_goal,
        "total_found":     len(results),
        "recommendations": results[:3],
    })


@app.route("/menu/<hall_slug>", methods=["GET"])
def get_hall_menu(hall_slug):
    meal, all_menus = _get_menus()
    matched_name = None
    for name, slug in YALE_DINING_HALLS.items():
        if slug == hall_slug or slug.rstrip("-college") == hall_slug.rstrip("-college"):
            matched_name = name
            break
    if not matched_name:
        for name in all_menus:
            if hall_slug.replace("-", " ").lower() in name.lower():
                matched_name = name
                break
    if not matched_name or matched_name not in all_menus:
        return jsonify({"error": f'No menu data for "{hall_slug}"',
                        "available": list(all_menus.keys())}), 404
    return jsonify({"hall": matched_name, "meal": meal,
                    "date": date.today().isoformat(), "menu": all_menus[matched_name]})


@app.route("/halls", methods=["GET"])
def get_halls():
    return jsonify({"halls": [{"name": n, "slug": s}
                               for n, s in YALE_DINING_HALLS.items()]})


@app.route("/health", methods=["GET"])
def health():
    ai_ready = bool(get_ai_client())
    return jsonify({
        "status": "ok",
        "meal":   get_current_meal(),
        "ai":     ai_ready,
        "model":  MODEL if ai_ready else None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

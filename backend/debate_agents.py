"""
debate_agents.py - Single-agent meal recommender for Yale dining halls.

The agent searches today's menu using BM25 RAG tools and recommends
the best meal options based on the user's goals and restrictions.
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

import anthropic

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
)

MODEL = os.environ.get("MINIMAX_MODEL", "abab6.5s-chat")


# ============================================================
# Tool Definitions (Anthropic tool_use format)
# ============================================================

TOOLS: List[dict] = [
    {
        "name": "search_menu",
        "description": (
            "Search all dining hall menus for food items matching a query. "
            "Returns the top-k most relevant items with nutritional info and dietary flags. "
            "Use this to find specific dishes, ingredients, or cuisine types."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g. 'grilled chicken', 'vegan pasta', 'high protein breakfast')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10, max 20)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "filter_by_dietary_need",
        "description": (
            "Find menu items that match a specific dietary restriction or label. "
            "Useful for vegan, vegetarian, gluten-free, halal, kosher filtering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "restriction": {
                    "type": "string",
                    "description": "Dietary restriction keyword (e.g. 'vegan', 'gluten-free', 'halal', 'vegetarian')",
                },
            },
            "required": ["restriction"],
        },
    },
    {
        "name": "compare_nutrition",
        "description": (
            "Look up and compare the nutritional content (calories, protein, carbs, fat, fiber) "
            "of specific food items."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of food item names to compare",
                },
            },
            "required": ["item_names"],
        },
    },
]


# ============================================================
# Tool Execution Engine
# ============================================================

def _fmt_item(item: dict) -> str:
    cal     = f"{int(item['calories'])} cal"    if item.get("calories") else "cal N/A"
    protein = f"{item['protein_g']}g protein"   if item.get("protein_g") else "protein N/A"
    carbs   = f"{item.get('carbs_g')}g carbs"   if item.get("carbs_g")   else ""
    fat     = f"{item.get('fat_g')}g fat"        if item.get("fat_g")     else ""
    flags   = ", ".join(item.get("dietary_flags", [])) or "no special flags"
    station = f" [{item.get('station', '')}]" if item.get("station") else ""
    parts   = [p for p in [cal, protein, carbs, fat] if p]
    return f"- {item['name']}{station}: {' | '.join(parts)} | flags: {flags}"


def execute_tool(tool_name: str, tool_input: dict, rag, all_menus: dict) -> str:
    """Route a tool call to the appropriate handler and return a string result."""

    if tool_name == "search_menu":
        query  = tool_input.get("query", "")
        top_k  = min(int(tool_input.get("top_k", 10)), 20)
        results = rag.search(query, top_k=top_k)
        if not results:
            return f"No results found for '{query}'"
        lines = [f"Search results for '{query}':"]
        for _, _, meta, _ in results:
            lines.append(_fmt_item(meta))
        return "\n".join(lines)

    if tool_name == "filter_by_dietary_need":
        restriction = tool_input.get("restriction", "").lower()
        matches = []
        for hall_name, menu in all_menus.items():
            for station, items in menu.items():
                for item in items:
                    flags_lower = " ".join(f.lower() for f in item.get("dietary_flags", []))
                    if restriction in flags_lower:
                        matches.append(_fmt_item({**item, "station": station}))
        if not matches:
            return f"No items found matching '{restriction}'"
        return f"Items matching '{restriction}':\n" + "\n".join(matches[:25])

    if tool_name == "compare_nutrition":
        item_names = tool_input.get("item_names", [])
        rows = []
        for name in item_names:
            results = rag.search(name, top_k=1)
            if results:
                _, _, meta, _ = results[0]
                rows.append(
                    f"  {meta['name']}:\n"
                    f"    Calories: {meta.get('calories') or 'N/A'}\n"
                    f"    Protein:  {meta.get('protein_g') or 'N/A'}g\n"
                    f"    Carbs:    {meta.get('carbs_g') or 'N/A'}g\n"
                    f"    Fat:      {meta.get('fat_g') or 'N/A'}g\n"
                    f"    Fiber:    {meta.get('fiber_g') or 'N/A'}g"
                )
        if not rows:
            return "No nutritional data found for those items."
        return "Nutritional Comparison:\n" + "\n".join(rows)

    return f"Unknown tool: {tool_name}"


# ============================================================
# Agentic Loop
# ============================================================

def run_agent(
    system_prompt: str,
    user_message: str,
    rag,
    all_menus: dict,
    max_turns: int = 6,
) -> str:
    """
    Core agentic loop: send message → execute tool calls → repeat until done.
    Returns the final text response from the model.
    """
    messages = [{"role": "user", "content": user_message}]
    final_text = ""

    for _ in range(max_turns):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            if "tool" in str(e).lower() or "function" in str(e).lower():
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=messages,
                )
            else:
                raise

        text_parts = []
        tool_uses  = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            if hasattr(block, "type") and block.type == "tool_use":
                tool_uses.append(block)

        final_text = "".join(text_parts)

        if response.stop_reason == "end_turn" or not tool_uses:
            return final_text

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            result_text = execute_tool(tu.name, tu.input, rag, all_menus)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tu.id,
                "content":     result_text,
            })
        messages.append({"role": "user", "content": tool_results})

    return final_text or "[Agent reached max turns]"


# ============================================================
# Recommender Agent
# ============================================================

def run_recommender(prefs: dict, rag, all_menus: dict, meal: str) -> str:
    """
    Single agent that searches today's menu and recommends the best
    meal options based on the user's goals and restrictions.
    """
    system = """You are a helpful Yale dining hall food advisor.
Your job: recommend the best meal options from today's dining menu based on the user's goals, restrictions, and preferences.

Rules:
- Use the search and filter tools to find REAL items available today
- Always respect dietary restrictions and allergies — these are non-negotiable
- Cite actual food names, calorie counts, and protein numbers
- Keep your recommendation focused and practical

Your response must follow this format:

🍽️  RECOMMENDED MEAL:
1. [Dish name] — [why it fits, with key nutrition stats]
2. [Dish name] — [why it fits, with key nutrition stats]
3. [Dish name] — [why it fits, with key nutrition stats]

💡 TIP: [One practical tip about eating at the dining hall today]

⚠️  NOTE: [Any allergy warnings or important caveats, or omit if none]"""

    prefs_block = (
        f"User dietary profile:\n"
        f"  Goal:         {prefs.get('goal', 'balanced meal')}\n"
        f"  Restrictions: {prefs.get('restrictions', 'none')}\n"
        f"  Allergies:    {prefs.get('allergies', 'none')}\n"
        f"  Preferences:  {prefs.get('preferences', 'no specific preference')}"
    )

    prompt = (
        f"{prefs_block}\n\n"
        f"Today's meal is {meal}. Use the tools to search the menu and recommend "
        f"the best dishes for this user. Be specific and practical."
    )

    return run_agent(system, prompt, rag, all_menus, max_turns=6)

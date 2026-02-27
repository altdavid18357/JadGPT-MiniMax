"""
menu_fetcher.py - Fetch live menu data from all Yale dining halls via Nutrislice API.
Uses parallel requests to minimize load time.
"""

import requests
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# All working Yale dining halls with their Nutrislice API slugs
YALE_DINING_HALLS = {
    "Benjamin Franklin": "benjamin-franklin-college",
    "Branford":          "branford-college",
    "Davenport":         "davenport-college",
    "Jonathan Edwards":  "jonathan-edwards-college",
    "Berkeley":          "berkeley-college",
    "Pierson":           "pierson-college",
    "Saybrook":          "saybrook-college",
    "Silliman":          "silliman-college",
    "Timothy Dwight":    "timothy-dwight-college",
    "Trumbull":          "trumbull-college",
    "Ezra Stiles":       "ezra-stiles-college",
    "Morse":             "morse-college",
}

BASE_URL = "https://yalehospitality.api.nutrislice.com/menu/api/weeks/school"


def get_current_meal() -> str:
    now = datetime.now().time()
    if now < datetime.strptime("11:00", "%H:%M").time():
        return "breakfast"
    elif now < datetime.strptime("14:30", "%H:%M").time():
        return "lunch"
    else:
        return "dinner"


def _fetch_raw(hall_slug: str, meal: str, date_str: str) -> dict:
    url = f"{BASE_URL}/{hall_slug}/menu-type/{meal}/{date_str}/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _parse_menu(raw: dict, target_date: str) -> Dict[str, List[dict]]:
    """Parse Nutrislice response into {station: [items]} for target_date."""
    menu = {}
    current_station = "General"

    for day in raw.get("days", []):
        if day.get("date") != target_date:
            continue
        for item in day.get("menu_items", []):
            if item.get("is_station_header") and item.get("text"):
                current_station = item["text"]
                menu.setdefault(current_station, [])
            elif item.get("food") and item["food"].get("name"):
                food = item["food"]
                nut = food.get("rounded_nutrition_info") or {}
                flags = [
                    icon["name"]
                    for icon in food.get("icons", {}).get("food_icons", [])
                    if icon.get("name")
                ]
                menu.setdefault(current_station, []).append({
                    "name":          food["name"],
                    "description":   food.get("description", ""),
                    "calories":      nut.get("calories"),
                    "protein_g":     nut.get("g_protein"),
                    "carbs_g":       nut.get("g_total_carb"),
                    "fat_g":         nut.get("g_total_fat"),
                    "fiber_g":       nut.get("g_dietary_fiber"),
                    "sodium_mg":     nut.get("mg_sodium"),
                    "dietary_flags": flags,
                })
    return menu


def _fetch_hall(args: Tuple) -> Tuple[str, Dict[str, List[dict]]]:
    hall_name, hall_slug, meal, date_str, target_date = args
    raw = _fetch_raw(hall_slug, meal, date_str)
    menu = _parse_menu(raw, target_date)
    return hall_name, menu


def fetch_all_menus(verbose: bool = True) -> Tuple[str, Dict[str, Dict[str, List[dict]]]]:
    """
    Fetch menus from all Yale dining halls in parallel.
    Returns (meal_name, {hall_name: {station: [items]}})
    """
    meal = get_current_meal()
    today = date.today()
    date_str = today.strftime("%Y/%m/%d")
    target_date = today.isoformat()

    if verbose:
        print(f"Fetching {meal} menus for {today.strftime('%B %d, %Y')} "
              f"from {len(YALE_DINING_HALLS)} dining halls...")

    tasks = [
        (name, slug, meal, date_str, target_date)
        for name, slug in YALE_DINING_HALLS.items()
    ]

    all_menus: Dict[str, Dict[str, List[dict]]] = {}

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fetch_hall, t): t[0] for t in tasks}
        for future in as_completed(futures):
            hall_name = futures[future]
            try:
                name, menu = future.result()
                if menu:
                    all_menus[name] = menu
                    total = sum(len(v) for v in menu.values())
                    if verbose:
                        print(f"  ✓ {name}: {total} items across {len(menu)} stations")
                else:
                    if verbose:
                        print(f"  ✗ {name}: no menu data")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {hall_name}: error - {e}")

    return meal, all_menus


def format_menu_text(hall_name: str, menu: Dict[str, List[dict]]) -> str:
    """Format a hall's menu as readable text (for context injection)."""
    lines = [f"=== {hall_name} Menu ==="]
    for station, items in menu.items():
        if not items:
            continue
        lines.append(f"\n[{station}]")
        for item in items:
            cal = f"{int(item['calories'])} cal" if item.get("calories") else "cal N/A"
            pro = f"{item['protein_g']}g protein" if item.get("protein_g") else "protein N/A"
            flags = ", ".join(item["dietary_flags"]) if item["dietary_flags"] else "none"
            lines.append(f"  - {item['name']}: {cal} | {pro} | {flags}")
    return "\n".join(lines)

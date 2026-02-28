"""
menu_fetcher.py - Fetch live menu data from all Yale dining halls via Nutrislice API.

Uses /menu/api/schools/ for dynamic hall discovery and real per-meal serving hours.
Falls back to hardcoded data if the schools API is unavailable.
"""

import requests
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

NUTRISLICE_BASE = "https://yalehospitality.api.nutrislice.com"
SCHOOLS_API_URL = f"{NUTRISLICE_BASE}/menu/api/schools/"

# Fallback hall list if the schools API fails
_FALLBACK_HALLS = {
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

# Keep for backward compatibility (used by main.py /halls endpoint)
YALE_DINING_HALLS = _FALLBACK_HALLS

# Module-level cache: populated on first call to fetch_schools()
_schools_cache: List[dict] = []


# ── Schools API ───────────────────────────────────────────────

def fetch_schools() -> List[dict]:
    """
    Fetch all Yale dining halls from the Nutrislice schools API.
    Returns a list of dicts with name, slug, meal_types (URL templates), and
    serving_hours (per-day windows for breakfast/lunch/dinner).
    Result is cached for the lifetime of the process.
    """
    global _schools_cache
    if _schools_cache:
        return _schools_cache

    try:
        r = requests.get(SCHOOLS_API_URL, timeout=10)
        r.raise_for_status()
        raw_schools = r.json()
    except Exception:
        return []

    schools = []
    for s in raw_schools:
        name = s.get("name", "")
        slug = s.get("slug", "")
        if not name or not slug:
            continue

        # Per-meal URL templates: {"breakfast": "/menu/api/weeks/school/…/{year}/{month}/{day}", …}
        meal_types: Dict[str, str] = {}
        for mt in s.get("active_menu_types", []):
            mt_name = mt.get("name", "").lower()   # "breakfast", "lunch", "dinner"
            template = mt.get("urls", {}).get("full_menu_by_date_api_url_template", "")
            if mt_name and template:
                meal_types[mt_name] = template

        # Per-meal serving windows: {"breakfast": {"mon": {"start": "08:00:00", "end": "11:00:00"}, …}, …}
        serving_hours: Dict[str, Dict[str, dict]] = {}
        for op in s.get("operating_days_by_menu_type", []):
            mt_name = op.get("menu_type_name", "").lower()
            if not mt_name:
                continue
            day_windows: Dict[str, dict] = {}
            for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
                if op.get(f"{day}_enabled"):
                    day_windows[day] = {
                        "start": op.get(f"{day}_start", ""),
                        "end":   op.get(f"{day}_end", ""),
                    }
            serving_hours[mt_name] = day_windows

        schools.append({
            "name":          name,
            "slug":          slug,
            "meal_types":    meal_types,
            "serving_hours": serving_hours,
        })

    _schools_cache = schools
    return schools


def get_hall_list() -> List[Dict[str, str]]:
    """Return [{name, slug}] for all dining halls (dynamic from API, fallback to hardcoded)."""
    schools = fetch_schools()
    if schools:
        return [{"name": s["name"], "slug": s["slug"]} for s in schools]
    return [{"name": k, "slug": v} for k, v in _FALLBACK_HALLS.items()]


# ── Meal timing ───────────────────────────────────────────────

def get_current_meal() -> str:
    """
    Return the current meal (breakfast/lunch/dinner) based on real serving hours
    from the Nutrislice schools API. Falls back to hardcoded windows if unavailable.
    """
    now = datetime.now()
    # strftime("%a") returns "Mon", "Tue", etc. — lowercase to match our keys
    day_abbr = now.strftime("%a").lower()
    t = now.hour * 60 + now.minute   # minutes since midnight

    schools = fetch_schools()
    if schools:
        hours = schools[0].get("serving_hours", {})
        for meal_name in ("breakfast", "lunch", "dinner"):
            window = hours.get(meal_name, {}).get(day_abbr)
            if not window:
                continue
            try:
                sh, sm = int(window["start"][:2]), int(window["start"][3:5])
                eh, em = int(window["end"][:2]),   int(window["end"][3:5])
                if (sh * 60 + sm) <= t < (eh * 60 + em):
                    return meal_name
            except (ValueError, KeyError):
                continue

    # Fallback: hardcoded Yale serving windows
    if t < 11 * 60 + 30:
        return "breakfast"
    elif t < 15 * 60:
        return "lunch"
    elif t < 20 * 60:
        return "dinner"
    else:
        return "breakfast"


# ── Menu fetching ─────────────────────────────────────────────

def _fetch_raw(url: str) -> dict:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


_BREAKFAST_SKIP_STATIONS = {"smartmeals", "smart meals"}

def _parse_menu(raw: dict, target_date: str, meal: str = "") -> Dict[str, List[dict]]:
    """Parse Nutrislice response into {station: [items]} for target_date."""
    menu: Dict[str, List[dict]] = {}
    current_station = "General"

    for day in raw.get("days", []):
        if day.get("date") != target_date:
            continue
        for item in day.get("menu_items", []):
            if item.get("is_station_header") and item.get("text"):
                current_station = item["text"]
                menu.setdefault(current_station, [])
            elif item.get("food") and item["food"].get("name"):
                # SmartMeals is a lunch/dinner-only offering — skip it at breakfast
                if meal == "breakfast" and current_station.lower().strip() in _BREAKFAST_SKIP_STATIONS:
                    continue
                food = item["food"]
                nut  = food.get("rounded_nutrition_info") or {}
                flags = [
                    icon["name"]
                    for icon in food.get("icons", {}).get("food_icons", [])
                    if icon.get("name")
                ]
                serving_size = (
                    nut.get("serving_size")
                    or food.get("serving_size_info")
                    or food.get("serving_size")
                    or ""
                )
                menu.setdefault(current_station, []).append({
                    "name":          food["name"],
                    "description":   food.get("description", ""),
                    "calories":      nut.get("calories"),
                    "protein_g":     nut.get("g_protein"),
                    "carbs_g":       nut.get("g_total_carb"),
                    "fat_g":         nut.get("g_total_fat"),
                    "fiber_g":       nut.get("g_dietary_fiber"),
                    "sodium_mg":     nut.get("mg_sodium"),
                    "serving_size":  serving_size,
                    "dietary_flags": flags,
                })
    return menu


def _fetch_hall(args: Tuple) -> Tuple[str, Dict[str, List[dict]]]:
    hall_name, url, target_date, meal = args
    raw  = _fetch_raw(url)
    menu = _parse_menu(raw, target_date, meal)
    return hall_name, menu


def fetch_all_menus(verbose: bool = True) -> Tuple[str, Dict[str, Dict[str, List[dict]]]]:
    """
    Fetch menus from all Yale dining halls in parallel using the schools API.
    Returns (meal_name, {hall_name: {station: [items]}})
    """
    meal       = get_current_meal()
    today      = date.today()
    target_date = today.isoformat()
    year, month, day = str(today.year), f"{today.month:02d}", f"{today.day:02d}"

    schools = fetch_schools()

    # Fall back to legacy slug-based approach if schools API unavailable
    if not schools:
        return _fetch_all_menus_legacy(meal, today, verbose)

    if verbose:
        print(f"Fetching {meal} menus for {today.strftime('%B %d, %Y')} "
              f"from {len(schools)} dining halls...")

    tasks = []
    for school in schools:
        template = school["meal_types"].get(meal, "")
        if not template:
            continue
        url = NUTRISLICE_BASE + template.format(year=year, month=month, day=day)
        tasks.append((school["name"], url, target_date, meal))

    all_menus: Dict[str, Dict[str, List[dict]]] = {}

    with ThreadPoolExecutor(max_workers=14) as pool:
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


def _fetch_all_menus_legacy(
    meal: str, today: date, verbose: bool
) -> Tuple[str, Dict[str, Dict[str, List[dict]]]]:
    """Fallback: fetch using hardcoded hall slugs if schools API is unavailable."""
    date_str    = today.strftime("%Y/%m/%d")
    target_date = today.isoformat()
    base        = f"{NUTRISLICE_BASE}/menu/api/weeks/school"

    if verbose:
        print(f"[fallback] Fetching {meal} menus from {len(_FALLBACK_HALLS)} halls...")

    tasks = [
        (name, f"{base}/{slug}/menu-type/{meal}/{date_str}/", target_date, meal)
        for name, slug in _FALLBACK_HALLS.items()
    ]

    all_menus: Dict[str, Dict[str, List[dict]]] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fetch_hall, t): t[0] for t in tasks}
        for future in as_completed(futures):
            try:
                name, menu = future.result()
                if menu:
                    all_menus[name] = menu
            except Exception:
                pass

    return meal, all_menus


# ── Formatting ────────────────────────────────────────────────

def format_menu_text(hall_name: str, menu: Dict[str, List[dict]]) -> str:
    """Format a hall's menu as readable text (for context injection)."""
    lines = [f"=== {hall_name} Menu ==="]
    for station, items in menu.items():
        if not items:
            continue
        lines.append(f"\n[{station}]")
        for item in items:
            cal   = f"{int(item['calories'])} cal" if item.get("calories") else "cal N/A"
            pro   = f"{item['protein_g']}g protein" if item.get("protein_g") else "protein N/A"
            flags = ", ".join(item["dietary_flags"]) if item["dietary_flags"] else "none"
            lines.append(f"  - {item['name']}: {cal} | {pro} | {flags}")
    return "\n".join(lines)

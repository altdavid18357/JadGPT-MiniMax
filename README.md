# BoolaBites — Yale Dining Recommender

An AI-powered meal planner for Yale dining halls. BoolaBites fetches live menus from all Yale residential college dining halls, then uses a MiniMax AI agent to build personalized meal plates that match your dietary restrictions, calorie goals, and food preferences.

---

## Features

- **Live menu data** — Pulls real-time menus from 12+ Yale dining halls via the Nutrislice API
- **AI meal planner** — MiniMax agent with tool use assembles portion-aware meal combinations
- **Dietary restriction enforcement** — Vegan, vegetarian, gluten-free, halal
- **Nutritional targeting** — Hits your daily calorie and protein goals (~1/3 per meal)
- **BM25 search** — Fast keyword retrieval without an external vector database
- **Multi-hall recommendations** — Picks the best dishes across all open dining halls
- **Web UI + CLI** — Browser interface and command-line interface both available

---

## Project Structure

```
JadGPT-MiniMax/
├── backend/
│   ├── main.py            # Flask API server (5 endpoints)
│   ├── menu_fetcher.py    # Nutrislice API integration
│   ├── debate_agents.py   # MiniMax AI agent with tool use
│   ├── rag_system.py      # BM25 search/retrieval system
│   ├── run_debate.py      # CLI interface
│   └── requirements.txt   # Python dependencies
├── frontend/
│   └── index.html         # Single-page web app (vanilla JS)
├── main.py                # Standalone menu fetcher script
└── LICENSE
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | Flask 3.1.0 |
| AI / LLM | MiniMax via Anthropic SDK (`abab6.5s-chat`) |
| Menu data | Nutrislice API (Yale Hospitality) |
| Search | Custom BM25 (no external DB) |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Python | 3.9+ |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/altdavid18357/JadGPT-MiniMax.git
cd JadGPT-MiniMax

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt
```

### 2. Configure environment variables

Create `backend/.env`:

```env
ANTHROPIC_API_KEY=your_minimax_api_key_here
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
```

> The Anthropic SDK is pointed at the MiniMax endpoint. Get your key from [MiniMax](https://api.minimax.io).

---

## Running the App

### Web app (recommended)

```bash
# Terminal 1 — start backend
cd backend
python main.py
# API server runs on http://localhost:5000

# Terminal 2 — open frontend
open frontend/index.html
# Or serve it with any static file server
```

### CLI

```bash
cd backend

python run_debate.py              # Interactive prompts
python run_debate.py --demo       # Preset: high-protein, meat-forward
python run_debate.py --demo-vegan # Preset: balanced vegan
python run_debate.py --meal lunch # Force a specific meal type
```

### Standalone menu fetcher

```bash
python main.py   # Fetches and prints the Benjamin Franklin College menu
```

---

## API Reference

Base URL: `http://localhost:5000`

### `GET /health`
Health check. Returns current meal type (breakfast/lunch/dinner).

---

### `GET /halls`
Lists all available Yale dining halls and their Nutrislice slugs.

**Response:**
```json
[
  { "name": "Benjamin Franklin", "slug": "benjamin-franklin" },
  ...
]
```

---

### `GET /menu/<hall_slug>`
Full menu for a specific dining hall, grouped by station.

---

### `GET|POST /recommend`
Rules-based meal recommendations filtered by dietary restrictions and ranked by protein content.

**Parameters** (query string or JSON body):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `restrictions` | string or array | `""` | Dietary restrictions (e.g. `"vegan,gluten-free"`) |
| `calorie_goal` | int | `2000` | Daily calorie target |
| `protein_goal` | int | `50` | Daily protein target (grams) |
| `allergies` | string | `""` | Allergens to avoid |
| `preferences` | string | `""` | Free-text food preferences |

**Response:**
```json
{
  "meal": "lunch",
  "date": "2026-02-27",
  "calorie_goal": 2000,
  "protein_goal": 50,
  "total_found": 45,
  "recommendations": [
    {
      "name": "Grilled Chicken",
      "calories": 320,
      "protein_g": 45,
      "carbs_g": 0,
      "fat_g": 12,
      "dietary_flags": ["gluten-free"],
      "dining_hall": "Benjamin Franklin",
      "station": "Grill"
    }
  ]
}
```

---

### `POST /agent/recommend`
AI-powered meal plan. The MiniMax agent uses tools to search and filter the menu, then assembles a portion-aware meal plate.

**Request body:** Same parameters as `/recommend`.

**Response:**
```json
{
  "meal": "lunch",
  "date": "2026-02-27",
  "meal_plan": "🍽️ YOUR MEAL PLATE:\n1. Grilled Chicken — 1 serving\n   📍 Benjamin Franklin · Grill\n   ...",
  "picks": [ { ...menu item objects... } ]
}
```

**Meal plan format:**
```
🍽️ YOUR MEAL PLATE:
1. [Dish] — [portion]
   [Hall] · [Station]
   Nutrition: [X] cal | [Y]g protein

COMBINED: [total] cal | [total]g protein
TIP: [Practical advice]
NOTE: [Allergy warning if relevant]
```

---

## How the AI Agent Works

1. **Menu fetching** — All dining hall menus are fetched in parallel (14 worker threads) from the Nutrislice API and cached for 10 minutes.
2. **BM25 indexing** — Every menu item is indexed by name, station, dietary flags, and nutrition descriptors.
3. **Agent loop** — The MiniMax agent is given three tools:
   - `search_menu` — keyword search over the BM25 index
   - `filter_by_dietary_need` — hard-filter by restriction
   - `compare_nutrition` — compare nutritional values
4. **Meal assembly** — The agent targets ~1/3 of daily calorie/protein goals, recommends 2–3 dish combinations, and specifies exact dining hall locations and stations.
5. **Hard constraints** — Dietary restrictions are never relaxed; the agent will not invent dishes not present in the menu.

---

## Dining Halls

Benjamin Franklin · Branford · Davenport · Jonathan Edwards · Berkeley · Pierson · Saybrook · Silliman · Timothy Dwight · Trumbull · Ezra Stiles · Morse

Hall availability is discovered dynamically from the Nutrislice schools API, with the above list as a fallback.

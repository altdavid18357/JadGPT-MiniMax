import requests
from datetime import date, datetime

NOW = datetime.now().time()
if NOW < datetime.strptime("11:00", "%H:%M").time():
    MEAL = "breakfast"
elif NOW < datetime.strptime("14:30", "%H:%M").time():
    MEAL = "lunch"
else:
    MEAL = "dinner"

TODAY = date.today().strftime("%Y/%m/%d")
URL = f"https://yalehospitality.api.nutrislice.com/menu/api/weeks/school/benjamin-franklin-college/menu-type/{MEAL}/{TODAY}/"

response = requests.get(URL)
data = response.json()

# Find today's menu items across all days in the week response
menu_by_station = {}
current_station = "General"

for day in data.get("days", []):
    if day.get("date") != date.today().isoformat():
        continue
    for item in day.get("menu_items", []):
        # Station headers define the category name
        if item.get("is_station_header") and item.get("text"):
            current_station = item["text"]
            if current_station not in menu_by_station:
                menu_by_station[current_station] = []
        # Food items have a non-null food object
        elif item.get("food") and item["food"].get("name"):
            food = item["food"]

            nutrition = food.get("rounded_nutrition_info", {})
            calories = nutrition.get("calories")
            protein = nutrition.get("g_protein")

            restrictions = [
                icon["name"]
                for icon in food.get("icons", {}).get("food_icons", [])
                if icon.get("name")
            ]

            if current_station not in menu_by_station:
                menu_by_station[current_station] = []
            menu_by_station[current_station].append({
                "name": food["name"],
                "calories": calories,
                "protein_g": protein,
                "dietary_flags": restrictions,
            })

# Print results
print(f"{MEAL.capitalize()} Menu for {date.today().strftime('%B %d, %Y')}:\n")
for station, foods in menu_by_station.items():
    if foods:
        print(f"  [{station}]")
        for food in foods:
            cal = f"{int(food['calories'])} cal" if food["calories"] is not None else "cal N/A"
            pro = f"{food['protein_g']}g protein" if food["protein_g"] is not None else "protein N/A"
            flags = ", ".join(food["dietary_flags"]) if food["dietary_flags"] else "none"
            print(f"    - {food['name']}")
            print(f"        {cal} | {pro} | allergens/diet: {flags}")
        print()

# app.py (Version 4.0 - Deterministic Ranking Engine)
import streamlit as st
import pandas as pd
import ast
import requests

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PantryChef AI",
    page_icon="🍳",
    layout="wide"
)

# --- LOAD DATA (MODEL IS NO LONGER NEEDED) ---
@st.cache_data
def load_data():
    """Loads the de-duplicated and cleaned recipe dataset from a URL."""
    # Ensure this URL points to the recipes_cleaned.csv you created with explore_data.py
    DATA_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v1.0.0/recipes_cleaned.csv"
    
    try:
        df = pd.read_csv(DATA_URL)
        df['ingredients_list'] = df['ingredients_list'].apply(ast.literal_eval)
        return df
    except Exception as e:
        st.error(f"Error loading recipe data from URL: {e}")
        return None

recipes_df = load_data()

# --- THE NEW RANKING ENGINE ---
def rank_recipes(user_ingredients, recipes_df):
    """
    Ranks recipes based on a deterministic scoring system, not a predictive model.
    """
    ranked_recipes = []
    
    for index, recipe in recipes_df.iterrows():
        recipe_ingredients = set(recipe['ingredients_list'])
        common_ingredients = user_ingredients.intersection(recipe_ingredients)
        missing_ingredients = recipe_ingredients - user_ingredients
        
        # --- SCORING LOGIC ---
        # 1. Primary Score: Heavily reward each matching ingredient.
        ingredient_match_score = len(common_ingredients) * 20
        
        # 2. Penalty Score: Heavily penalize each missing ingredient.
        # This is the most important rule. A recipe you can't make is a bad match.
        missing_penalty = len(missing_ingredients) * 50
        
        # 3. Bonus for having all ingredients
        perfect_match_bonus = 50 if len(missing_ingredients) == 0 else 0
        
        # 4. Tie-breaker Score: Use rating as a small bonus.
        # This only matters if two recipes have a similar match score.
        popularity_bonus = recipe['rating'] * 2
        
        # --- FINAL SCORE ---
        final_score = ingredient_match_score - missing_penalty + perfect_match_bonus + popularity_bonus
        
        # We only want to see recipes that have at least some relevance
        if final_score > 0:
            ranked_recipes.append({
                'recipe': recipe,
                'score': final_score,
                'missing_count': len(missing_ingredients)
            })

    # Sort the list of dictionaries by score, descending
    sorted_recipes = sorted(ranked_recipes, key=lambda x: x['score'], reverse=True)
    
    return sorted_recipes

# --- UI & APP LOGIC ---
st.title("🍳 PantryChef AI")
st.markdown("Tired of wondering what to cook? **Enter what you have below, separated by commas,** and let our AI find the perfect recipe for you!")

user_input = st.text_area(
    "Enter the ingredients you have in your pantry:",
    "spaghetti, canned tomatoes, ground beef, onion, garlic, olive oil, red wine, basil, parmesan cheese",
    height=150,
    placeholder="e.g., flour, sugar, eggs, butter, milk..."
)

if st.button("Find Recipes!", type="primary", use_container_width=True):
    if recipes_df is None:
        st.error("The application's recipe data failed to load. Please refresh.")
    elif not user_input.strip():
        st.warning("Please enter some ingredients to get started.")
    else:
        user_ingredients = set([item.strip().lower() for item in user_input.split(',')])
        st.write(f"**Analyzing your {len(user_ingredients)} ingredients...**")
        
        # Use our new ranking engine
        sorted_recipes = rank_recipes(user_ingredients, recipes_df)
        
        if not sorted_recipes:
            st.error("Couldn't find any good recipe matches. Try adding more core ingredients.")
        else:
            st.success(f"🎉 Found and ranked {len(sorted_recipes)} potential recipes! Here are your top 5 recommendations:")
            
            # --- Display Results ---
            for result in sorted_recipes[:5]:
                recipe = result['recipe']
                score = result['score']
                missing_count = result['missing_count']
                
                missing_ingredients = set(recipe['ingredients_list']) - user_ingredients
                
                st.subheader(recipe['recipe_name'])
                
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.metric("Match Score", f"{int(score)}")
                with col2:
                    st.metric("Missing Ingredients", f"{missing_count}")

                if missing_ingredients:
                    st.warning(f"**You might need:** {', '.join(sorted(list(missing_ingredients)))}")
                else:
                    st.success("✅ You have all the ingredients for this recipe!")
                
                with st.expander("See full ingredient list and details"):
                    st.markdown(f"**Popularity Rating:** {recipe['rating']:.2f} / 5.0")
                    st.markdown(f"**Complexity:** {recipe['recipe_complexity']} steps")
                    st.markdown("**Full Ingredient List:**")
                    st.markdown("\n".join(f"- {ing}" for ing in recipe['ingredients_list']))
                
                st.divider()

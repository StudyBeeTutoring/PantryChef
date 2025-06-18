# app.py (Upgraded Cloud Deployment Version)
import streamlit as st
import pandas as pd
import joblib
import ast
import requests

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PantryChef AI",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- LOAD ASSETS (FROM GITHUB RELEASES URLS) ---
# Caching is essential for performance, so Streamlit doesn't re-download these large files on every interaction.

@st.cache_data
def load_data():
    """Loads the de-duplicated and cleaned recipe dataset from a raw GitHub URL."""
    # --- IMPORTANT: PASTE THE GITHUB RELEASE URL FOR YOUR CSV FILE HERE ---
    DATA_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v1.5.0/recipes_cleaned.csv"
    
    try:
        df = pd.read_csv(DATA_URL)
        df['ingredients_list'] = df['ingredients_list'].apply(ast.literal_eval)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        st.error(f"Please check the URL in your app.py: {DATA_URL}")
        return None

@st.cache_resource
def load_model():
    """Loads the upgraded trained model and columns from raw GitHub URLs."""
    # --- IMPORTANT: PASTE THE GITHUB RELEASE URLS FOR YOUR NEW MODEL FILES HERE ---
    MODEL_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v1.5.0/pantry_chef_model.joblib"
    COLUMNS_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v1.5.0/model_columns.joblib"
    
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open("pantry_chef_model.joblib", "wb") as f:
            f.write(response.content)
            
        response = requests.get(COLUMNS_URL)
        response.raise_for_status()
        with open("model_columns.joblib", "wb") as f:
            f.write(response.content)

        model = joblib.load("pantry_chef_model.joblib")
        model_cols = joblib.load("model_columns.joblib")
        
        return model, model_cols
    except Exception as e:
        st.error(f"Error loading model from URL: {e}")
        st.error(f"Please check the URLs in your app.py: {MODEL_URL}, {COLUMNS_URL}")
        return None, None

recipes_df = load_data()
model, model_columns = load_model()

# --- UI & APP LOGIC ---
st.title("🍳 PantryChef AI")
st.markdown("Got ingredients? Don't know what to cook? **Enter what you have below, separated by commas,** and let AI find the best recipes for you!")

user_input = st.text_area(
    "Enter the ingredients you have in your pantry:",
    "chicken breast, onion, garlic, olive oil, rice, salt, pepper, carrots, soy sauce",
    height=150,
    placeholder="e.g., flour, sugar, eggs, butter, milk..."
)

if st.button("Find Recipes!", type="primary", use_container_width=True):
    if recipes_df is None or model is None:
        st.error("The application's data or model failed to load. Please check the deployment settings.")
    elif not user_input.strip():
        st.warning("Please enter some ingredients to get started.")
    else:
        user_ingredients = set([item.strip().lower() for item in user_input.split(',')])
        st.write(f"**Searching for recipes with your {len(user_ingredients)} ingredients...**")
        
        # --- UPGRADED FILTERING LOGIC ---
        candidate_recipes = []
        for index, recipe in recipes_df.iterrows():
            recipe_ingredients = set(recipe['ingredients_list'])
            common_ingredients = user_ingredients.intersection(recipe_ingredients)
            
            # Stricter Rule: Only consider recipes where we have at least 25% of the ingredients
            match_percentage = len(common_ingredients) / len(recipe_ingredients) if len(recipe_ingredients) > 0 else 0
            if match_percentage >= 0.25:
                candidate_recipes.append(recipe)
        
        if not candidate_recipes:
            st.error("Couldn't find a good match. Try adding more common items like 'salt', 'pepper', or 'olive oil'.")
        else:
            # --- UPGRADED RANKING LOGIC ---
            candidates_df = pd.DataFrame(candidate_recipes)

            # This function must now calculate the NEW features our upgraded model was trained on
            def calculate_features(row):
                recipe_ingredients = set(row['ingredients_list'])
                common_ingredients = user_ingredients.intersection(recipe_ingredients)
                
                # These features must match the new model's training columns
                common_count = len(common_ingredients)
                match_pct = common_count / len(recipe_ingredients) if len(recipe_ingredients) > 0 else 0
                
                return pd.Series([common_count, match_pct, row['recipe_complexity'], row['rating']])
            
            features_df = candidates_df.apply(calculate_features, axis=1)
            features_df.columns = model_columns
            
            predictions = model.predict(features_df)
            candidates_df['predicted_score'] = predictions

            sorted_recipes = candidates_df.sort_values(by='predicted_score', ascending=False)
            
            st.success(f"🎉 Found and ranked {len(sorted_recipes)} potential recipes! Here are your top 5 recommendations:")
            
            # --- Display Results ---
            for index, recipe in sorted_recipes.head(5).iterrows():
                recipe_ingredients = set(recipe['ingredients_list'])
                missing_ingredients = recipe_ingredients - user_ingredients
                
                st.subheader(recipe['recipe_name'])
                
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.metric("Match Quality", f"{recipe['predicted_score']:.0f}")
                with col2:
                    st.metric("Missing Ingredients", f"{len(missing_ingredients)}")

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

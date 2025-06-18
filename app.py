# app.py (Professional Overhaul Cloud Version)
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
@st.cache_data
def load_data():
    """Loads the de-duplicated and cleaned recipe dataset from a raw GitHub URL."""
    DATA_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v2.0.0/recipes_cleaned.csv"
    
    try:
        df = pd.read_csv(DATA_URL)
        df['ingredients_list'] = df['ingredients_list'].apply(ast.literal_eval)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        st.error(f"Please check the URL in your app.py: {DATA_URL}")
        return None

@st.cache_resource
def load_model_assets():
    """Loads the model, scaler, and columns from raw GitHub URLs."""
    MODEL_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v2.0.0/pantry_chef_model.joblib"
    COLUMNS_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v2.0.0/model_columns.joblib"
    SCALER_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v2.0.0/scaler.joblib"
    
    try:
        # Download model file
        response_model = requests.get(MODEL_URL)
        response_model.raise_for_status()
        with open("pantry_chef_model.joblib", "wb") as f:
            f.write(response_model.content)
            
        # Download columns file
        response_cols = requests.get(COLUMNS_URL)
        response_cols.raise_for_status()
        with open("model_columns.joblib", "wb") as f:
            f.write(response_cols.content)

        # Download scaler file
        response_scaler = requests.get(SCALER_URL)
        response_scaler.raise_for_status()
        with open("scaler.joblib", "wb") as f:
            f.write(response_scaler.content)

        # Load the assets from the temporary local files
        model = joblib.load("pantry_chef_model.joblib")
        model_cols = joblib.load("model_columns.joblib")
        scaler = joblib.load("scaler.joblib")
        
        return model, model_cols, scaler
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.error(f"Please check the URLs in your app.py for the model, columns, and scaler files.")
        return None, None, None

# Load all assets at the start of the app
recipes_df = load_data()
model, model_columns, scaler = load_model_assets()

# --- UI & APP LOGIC ---
st.title("🍳 PantryChef AI")
st.markdown("Tired of wondering what to cook? **Enter your ingredients below, separated by commas,** and let our AI find the perfect recipe for you!")

user_input = st.text_area(
    "Enter the ingredients you have in your pantry:",
    "chicken breast, onion, garlic, olive oil, rice, salt, pepper, carrots, soy sauce",
    height=150,
    placeholder="e.g., flour, sugar, eggs, butter, milk..."
)

if st.button("Find Recipes!", type="primary", use_container_width=True):
    if recipes_df is None or model is None or scaler is None:
        st.error("The application's core components failed to load. Please refresh or contact support.")
    elif not user_input.strip():
        st.warning("Please enter some ingredients to get started.")
    else:
        user_ingredients = set([item.strip().lower() for item in user_input.split(',')])
        st.write(f"**Analyzing your {len(user_ingredients)} ingredients...**")
        
        candidate_recipes = []
        for index, recipe in recipes_df.iterrows():
            recipe_ingredients = set(recipe['ingredients_list'])
            common_ingredients = user_ingredients.intersection(recipe_ingredients)
            
            # Stricter Rule: Only consider recipes where user has at least 25% of the ingredients
            match_percentage = len(common_ingredients) / len(recipe_ingredients) if len(recipe_ingredients) > 0 else 0
            if match_percentage >= 0.25:
                candidate_recipes.append(recipe)
        
        if not candidate_recipes:
            st.error("Couldn't find a good match. Try adding more common items like 'salt', 'pepper', or 'olive oil'.")
        else:
            candidates_df = pd.DataFrame(candidate_recipes)

            def calculate_features(row):
                recipe_ingredients = set(row['ingredients_list'])
                common_ingredients = user_ingredients.intersection(recipe_ingredients)
                
                common_count = len(common_ingredients)
                match_pct = common_count / len(recipe_ingredients) if len(recipe_ingredients) > 0 else 0
                
                return pd.Series([common_count, match_pct, row['recipe_complexity'], row['rating']])
            
            features_df = candidates_df.apply(calculate_features, axis=1)
            features_df.columns = model_columns
            
            # --- CRUCIAL STEP: SCALE THE FEATURES ---
            # The model was trained on scaled data, so we must scale live data too.
            features_scaled = scaler.transform(features_df)
            
            # Predict using the SCALED features
            predictions = model.predict(features_scaled)
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
                    st.metric("Match Score", f"{int(recipe['predicted_score'])}")
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

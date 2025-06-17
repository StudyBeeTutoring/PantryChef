# app.py (Cloud Deployment Version)
import streamlit as st
import pandas as pd
import joblib
import ast
import requests # Needed to download files from URLs

# --- CONFIGURATION ---
# Use st.set_page_config() as the first Streamlit command in your script.
st.set_page_config(
    page_title="PantryChef AI",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- LOAD ASSETS (FROM GITHUB RELEASES URLS) ---

# This section is modified to download your data and model from the cloud.
# Caching is essential for performance, so Streamlit doesn't re-download these large files on every interaction.

@st.cache_data
def load_data():
    """Loads the cleaned recipe dataset from a raw GitHub URL."""
    # --- IMPORTANT: PASTE THE GITHUB RELEASE URL FOR YOUR CSV FILE HERE ---
    DATA_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v1.0.0/recipes_cleaned.csv"
    
    try:
        df = pd.read_csv(DATA_URL)
        # The 'ingredients_list' column was saved as a string, so we convert it back to a list
        df['ingredients_list'] = df['ingredients_list'].apply(ast.literal_eval)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        st.error(f"Please check the URL in your app.py: {DATA_URL}")
        return None

@st.cache_resource
def load_model():
    """Loads the trained model and columns from raw GitHub URLs."""
    # --- IMPORTANT: PASTE THE GITHUB RELEASE URLS FOR YOUR MODEL FILES HERE ---
    MODEL_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v1.0.0/pantry_chef_model.joblib"
    COLUMNS_URL = "https://github.com/StudyBeeTutoring/PantryChef/releases/download/v1.0.0/model_columns.joblib"
    
    try:
        # Download the model file from the URL to a temporary local path
        response = requests.get(MODEL_URL)
        response.raise_for_status() # Raise an exception for bad status codes
        with open("pantry_chef_model.joblib", "wb") as f:
            f.write(response.content)
            
        # Download the columns file
        response = requests.get(COLUMNS_URL)
        response.raise_for_status()
        with open("model_columns.joblib", "wb") as f:
            f.write(response.content)

        # Now load them from the local files
        model = joblib.load("pantry_chef_model.joblib")
        model_cols = joblib.load("model_columns.joblib")
        
        return model, model_cols
    except Exception as e:
        st.error(f"Error loading model from URL: {e}")
        st.error(f"Please check the URLs in your app.py: {MODEL_URL}, {COLUMNS_URL}")
        return None, None

# Load all the necessary assets at the start of the app
recipes_df = load_data()
model, model_columns = load_model()


# --- UI & APP LOGIC ---

st.title("🍳 PantryChef AI")
st.markdown("Got ingredients? Don't know what to cook? **Enter what you have below, separated by commas,** and let AI find the best recipes for you!")

# User Input Text Area
user_input = st.text_area(
    "Enter the ingredients you have in your pantry:",
    "chicken breast, rice, onion, garlic, olive oil, soy sauce, carrots, peas",
    height=150,
    placeholder="e.g., flour, sugar, eggs, butter, milk..."
)

if st.button("Find Recipes!", type="primary", use_container_width=True):
    # Check if assets are loaded and user has provided input
    if recipes_df is None or model is None:
        st.error("The application's data or model failed to load. Please contact the administrator.")
    elif not user_input.strip():
        st.warning("Please enter some ingredients to get started.")
    else:
        # 1. Parse User Input: Convert the user's string into a clean set of lower-case ingredients.
        user_ingredients = set([item.strip().lower() for item in user_input.split(',')])
        
        st.write(f"**Searching for recipes with your {len(user_ingredients)} ingredients...**")
        
        # 2. Filter for Candidate Recipes: Find all recipes where the user has at least one ingredient.
        candidate_recipes = []
        for index, recipe in recipes_df.iterrows():
            recipe_ingredients = set(recipe['ingredients_list'])
            if user_ingredients.intersection(recipe_ingredients):
                candidate_recipes.append(recipe)
        
        if not candidate_recipes:
            st.error("Couldn't find any recipes with your ingredients. Try adding more common items like 'salt', 'pepper', or 'olive oil'.")
        else:
            # 3. Rank Candidates with the ML Model
            candidates_df = pd.DataFrame(candidate_recipes)

            # This function calculates the features for each candidate recipe based on the user's pantry
            def calculate_features(row):
                recipe_ingredients = set(row['ingredients_list'])
                common_ingredients = user_ingredients.intersection(recipe_ingredients)

                match_pct = len(common_ingredients) / len(recipe_ingredients)
                pantry_util = len(common_ingredients) / len(user_ingredients)
                missing_count = len(recipe_ingredients) - len(common_ingredients)
                
                return pd.Series([match_pct, pantry_util, missing_count, row['recipe_complexity'], row['rating']])
            
            # Create a DataFrame of features for all candidates at once
            features_df = candidates_df.apply(calculate_features, axis=1)
            features_df.columns = model_columns
            
            # Predict a "Match Score" for every candidate recipe using the trained model
            predictions = model.predict(features_df)
            candidates_df['predicted_score'] = predictions

            # Sort the recipes by the predicted score to find the best matches
            sorted_recipes = candidates_df.sort_values(by='predicted_score', ascending=False)
            
            st.success(f"🎉 Found and ranked {len(sorted_recipes)} recipes! Here are your top 5 recommendations:")
            
            # 4. Display Results in a clean, user-friendly format
            for index, recipe in sorted_recipes.head(5).iterrows():
                recipe_ingredients = set(recipe['ingredients_list'])
                missing_ingredients = recipe_ingredients - user_ingredients
                
                st.subheader(recipe['recipe_name'])
                
                col1, col2 = st.columns([1, 1.5]) # Adjust column widths
                with col1:
                    # Display the predicted score as a "Match Quality" metric
                    st.metric("Match Quality", f"{recipe['predicted_score']:.0f}/100")
                with col2:
                    # Show how many ingredients are missing
                    st.metric("Missing Ingredients", f"{len(missing_ingredients)}")

                # If there are missing ingredients, show them in a warning box
                if missing_ingredients:
                    st.warning(f"**You might need:** {', '.join(sorted(list(missing_ingredients)))}")
                else:
                    st.success("✅ You have all the ingredients for this recipe!")
                
                # Use an expander to neatly hide the full recipe details
                with st.expander("See full ingredient list and details"):
                    st.markdown(f"**Popularity Rating:** {recipe['rating']:.2f} / 5.0")
                    st.markdown(f"**Complexity:** {recipe['recipe_complexity']} steps")
                    st.markdown("**Full Ingredient List:**")
                    st.markdown("\n".join(f"- {ing}" for ing in recipe['ingredients_list']))
                
                st.divider() # Add a line between recipe recommendations

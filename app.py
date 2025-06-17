# app.py
import streamlit as st
import pandas as pd
import joblib
import ast

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PantryChef AI",
    page_icon="🍳",
    layout="wide"
)


# --- LOAD ASSETS ---
# Caching the data and model loading for better performance
@st.cache_data
def load_data():
    """Loads the cleaned recipe dataset."""
    try:
        df = pd.read_csv("recipes_cleaned.csv")
        # Convert the ingredients string back to a list
        df['ingredients_list'] = df['ingredients_list'].apply(ast.literal_eval)
        return df
    except FileNotFoundError:
        st.error("Error: 'recipes_cleaned.csv' not found. Please run Phase 1 script first.")
        return None


@st.cache_resource
def load_model():
    """Loads the trained machine learning model and its columns."""
    try:
        model = joblib.load("pantry_chef_model.joblib")
        model_cols = joblib.load("model_columns.joblib")
        return model, model_cols
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run Phase 2 script first.")
        return None, None


# Load all the necessary assets at the start
recipes_df = load_data()
model, model_columns = load_model()

# --- UI & APP LOGIC ---
st.title("🍳 PantryChef AI")
st.markdown(
    "Got ingredients? Don't know what to cook? **Enter your ingredients below, separated by commas,** and let our AI find the best recipes for you!")

# User Input
user_input = st.text_area(
    "Enter the ingredients you have in your pantry:",
    "chicken breast, rice, onion, garlic, olive oil, soy sauce, carrots, peas",
    height=150
)

if st.button("Find Recipes!", type="primary"):
    if recipes_df is None or model is None:
        st.warning("Please ensure the data and model files are available before searching.")
    elif not user_input.strip():
        st.warning("Please enter some ingredients.")
    else:
        # 1. Parse User Input
        # Convert input string to a clean set of ingredients
        user_ingredients = set([item.strip().lower() for item in user_input.split(',')])

        st.write(f"**Searching for recipes with your {len(user_ingredients)} ingredients...**")

        # 2. Filter for Candidate Recipes
        # Find recipes where the user has at least one matching ingredient to start
        candidate_recipes = []
        for index, recipe in recipes_df.iterrows():
            recipe_ingredients = set(recipe['ingredients_list'])
            common_ingredients = user_ingredients.intersection(recipe_ingredients)

            # We only consider a recipe if the user has at least one ingredient for it
            if len(common_ingredients) > 0:
                candidate_recipes.append(recipe)

        if not candidate_recipes:
            st.error(
                "Couldn't find any recipes with your ingredients. Try adding more common items like 'salt', 'pepper', or 'olive oil'.")
        else:
            # 3. Rank Candidates with the ML Model
            ranked_recipes = []

            # Create a DataFrame from the candidate recipes
            candidates_df = pd.DataFrame(candidate_recipes)


            # Calculate features for ALL candidates at once for efficiency
            def calculate_features(row):
                recipe_ingredients = set(row['ingredients_list'])
                common_ingredients = user_ingredients.intersection(recipe_ingredients)

                match_pct = len(common_ingredients) / len(recipe_ingredients)
                pantry_util = len(common_ingredients) / len(user_ingredients)
                missing_count = len(recipe_ingredients) - len(common_ingredients)

                return pd.Series([match_pct, pantry_util, missing_count, row['recipe_complexity'], row['rating']])


            # Apply the function and ensure the columns match the model's training columns
            features_df = candidates_df.apply(calculate_features, axis=1)
            features_df.columns = model_columns

            # Use the trained model to predict a "Match Score" for every candidate
            predictions = model.predict(features_df)
            candidates_df['predicted_score'] = predictions

            # Sort the recipes by the predicted score in descending order
            sorted_recipes = candidates_df.sort_values(by='predicted_score', ascending=False)

            st.success(f"🎉 Found and ranked {len(sorted_recipes)} recipes for you! Here are the top 5:")

            # 4. Display Results
            for index, recipe in sorted_recipes.head(5).iterrows():
                recipe_ingredients = set(recipe['ingredients_list'])
                missing_ingredients = recipe_ingredients - user_ingredients

                st.subheader(recipe['recipe_name'])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Match Score", f"{recipe['predicted_score']:.0f}/100")
                with col2:
                    st.metric("Missing Ingredients", f"{len(missing_ingredients)}")

                if missing_ingredients:
                    st.warning(f"You are missing: {', '.join(sorted(list(missing_ingredients)))}")
                else:
                    st.success("You have all the ingredients for this recipe!")

                with st.expander("See full ingredient list and details"):
                    st.markdown(f"**Popularity Rating:** {recipe['rating']:.2f}/5.0")
                    st.markdown(f"**Complexity:** {recipe['recipe_complexity']} steps")
                    st.markdown("**Full Ingredient List:**")
                    st.markdown("\n".join(f"- {ing}" for ing in recipe['ingredients_list']))
"""FastAPI app for Wiki Fraud Detection phase 1 model."""

from typing import List, Dict, Any
import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.predict_pipeline import Phase1PredictPipeline

app = FastAPI(title="Wiki Fraud Detection")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Phase1PredictPipeline()

# Load category dictionary
CATEGORIES_FILE = "data/processed/wiki/categories_api.json"
categories_data = None

def load_categories():
    """Load categories from JSON file."""
    global categories_data
    if categories_data is None:
        try:
            print(f"Looking for categories file at: {CATEGORIES_FILE}")
            print(f"File exists: {os.path.exists(CATEGORIES_FILE)}")
            print(f"Current working directory: {os.getcwd()}")
            
            if os.path.exists(CATEGORIES_FILE):
                print("Loading categories from JSON file...")
                with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
                    categories_data = json.load(f)
                print(f"Loaded {categories_data.get('total_count', 0)} categories")
            else:
                # Fallback: load from pickle and create JSON
                category_dict_path = "data/processed/wiki/category_dict.pkl"
                print(f"JSON not found, trying pickle at: {category_dict_path}")
                print(f"Pickle exists: {os.path.exists(category_dict_path)}")
                
                if os.path.exists(category_dict_path):
                    print("Loading from pickle...")
                    import pickle
                    with open(category_dict_path, "rb") as f:
                        category_dict = pickle.load(f)
                    
                    # Remove 'no_cat' metadata key
                    api_categories = {k: v for k, v in category_dict.items() if k != 'no_cat'}
                    
                    categories_data = {
                        "categories": [
                            {"id": str(category_id), "name": category_name, "display": f"{category_id} - {category_name}"}
                            for category_name, category_id in sorted(api_categories.items(), key=lambda x: x[1])
                        ],
                        "total_count": len(api_categories)
                    }
                    print(f"Created categories data with {len(api_categories)} categories")
                else:
                    print("No category files found, using empty data")
                    categories_data = {"categories": [], "total_count": 0}
        except Exception as e:
            print(f"Error loading categories: {e}")
            categories_data = {"categories": [], "total_count": 0}
    
    return categories_data


class UserEdits(BaseModel):
    edit_sequence: List[List[int]]
    rev_time: List[str]


@app.get("/categories")
def get_categories(search: str = "", limit: int = 50):
    """Get categories with optional search and limit."""
    categories_data = load_categories()
    categories = categories_data["categories"]
    
    if search:
        # Filter categories by search term (case insensitive)
        search_lower = search.lower()
        categories = [
            cat for cat in categories 
            if search_lower in cat["name"].lower()
        ]
    
    # Limit results
    if limit > 0:
        categories = categories[:limit]
    
    return {
        "categories": categories,
        "total_found": len(categories),
        "total_available": categories_data["total_count"]
    }

@app.get("/categories/random")
def get_random_categories(count: int = 10):
    """Get random categories for demo purposes."""
    import random
    categories_data = load_categories()
    categories = categories_data["categories"]
    
    if len(categories) == 0:
        return {"categories": [], "count": 0}
    
    # Get random sample
    sample_size = min(count, len(categories))
    random_categories = random.sample(categories, sample_size)
    
    return {
        "categories": random_categories,
        "count": len(random_categories)
    }

@app.get("/categories/domain/{domain}")
def get_domain_categories(domain: str, limit: int = 50):
    """Get categories by domain (e.g., american, academic, entertainment)."""
    categories_data = load_categories()
    all_categories = categories_data["categories"]
    
    # Domain-specific filtering
    domain_filters = {
        'american': lambda cat: 'american' in cat['name'].lower(),
        'british': lambda cat: 'british' in cat['name'].lower(),
        'academic': lambda cat: any(keyword in cat['name'].lower() for keyword in 
                                  ['university', 'college', 'school', 'education', 'academic', 
                                   'science', 'mathematics', 'physics', 'chemistry', 'biology']),
        'entertainment': lambda cat: any(keyword in cat['name'].lower() for keyword in
                                       ['film', 'movie', 'television', 'tv', 'music', 'album',
                                        'actor', 'actress', 'singer', 'musician', 'band']),
        'sports': lambda cat: any(keyword in cat['name'].lower() for keyword in
                                ['sport', 'athlete', 'player', 'team', 'game', 'olympic']),
        'technology': lambda cat: any(keyword in cat['name'].lower() for keyword in
                                    ['technology', 'computer', 'software', 'internet', 'web']),
        'medical': lambda cat: any(keyword in cat['name'].lower() for keyword in
                                 ['medical', 'health', 'hospital', 'disease', 'medicine']),
        'history': lambda cat: any(keyword in cat['name'].lower() for keyword in
                                 ['history', 'historical', 'ancient', 'medieval', 'war']),
        'geography': lambda cat: any(keyword in cat['name'].lower() for keyword in
                                   ['geography', 'cities', 'countries', 'states', 'regions']),
        'wikipedia_meta': lambda cat: any(keyword in cat['name'].lower() for keyword in
                                        ['articles', 'templates', 'stub', 'wikipedians'])
    }
    
    if domain.lower() not in domain_filters:
        return {
            "error": f"Domain '{domain}' not supported",
            "available_domains": list(domain_filters.keys()),
            "categories": [],
            "count": 0
        }
    
    # Filter categories by domain
    filter_func = domain_filters[domain.lower()]
    filtered_categories = [cat for cat in all_categories if filter_func(cat)]
    
    # Limit results
    if limit > 0:
        filtered_categories = filtered_categories[:limit]
    
    return {
        "domain": domain,
        "categories": filtered_categories,
        "count": len(filtered_categories),
        "total_available": len([cat for cat in all_categories if filter_func(cat)])
    }

@app.post("/predict")
def predict(edits: UserEdits):
    """Return vandalism probability for a set of user edits."""
    try:
        df = pd.DataFrame([edits.model_dump()])
        df["total_edits"] = df["edit_sequence"].apply(len)
        results = pipeline.predict(df)
        
        # Extract data for the first (and only) sample
        probability_percentage = float(results['probability_percentages'][0])
        label = results['labels'][0]
        raw_probability = float(results['raw_probabilities'][0])
        
        return {
            "prediction": raw_probability,  # Keep for backward compatibility
            "probability_percentage": probability_percentage,
            "label": label,
            "raw_probability": raw_probability
        }
    except Exception as exc:
        print(f"Exception: {exc}")  # Add this line for debugging
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/")
def root():
    return {"message": "Wiki Fraud Detection API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

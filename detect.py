"""
detect.py - Road Hazard Data Utility & Simulation

This script contains:
1. Utility functions for saving hazard metadata to hazards.json
2. Simulation logic for generating random hazards (for testing)
"""

import json
import random
import os
from datetime import datetime

# Global path to the hazards file
HAZARDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hazards.json")

def save_hazard(hazard_type, latitude, longitude, confidence):
    """
    Saves a new hazard to the JSON file safely.
    This is called by app.py when YOLO detects a pothole.
    """
    
    # Create the hazard object
    new_hazard = {
        "hazard_type": hazard_type,
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "confidence": round(float(confidence), 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Load existing hazards
    hazards = []
    if os.path.exists(HAZARDS_FILE):
        try:
            with open(HAZARDS_FILE, "r") as f:
                content = f.read()
                if content:
                    hazards = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # If file is corrupt or empty, start fresh
            hazards = []

    # Append new data
    hazards.append(new_hazard)

    # Save back to file
    with open(HAZARDS_FILE, "w") as f:
        json.dump(hazards, f, indent=2)
    
    return new_hazard

def generate_simulated_hazard():
    """
    Generates a single random hazard entry for simulation/testing.
    """
    h_type = random.choice(["pothole", "construction", "debris"])
    # Near Tiruchirappalli base coordinates
    lat = 10.7905 + random.uniform(-0.005, 0.005)
    lon = 78.7047 + random.uniform(-0.005, 0.005)
    conf = random.uniform(0.5, 0.95)
    
    return save_hazard(h_type, lat, lon, conf)

# ---- Main Execution (Simulation Mode) ----
if __name__ == "__main__":
    print("🔍 Generating a simulated hazard for testing...")
    hazard = generate_simulated_hazard()
    print(f"✅ Saved: {hazard['hazard_type']} at {hazard['latitude']}, {hazard['longitude']}")

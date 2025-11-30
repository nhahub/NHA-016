from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import math
import os

app = Flask(__name__)

# --- Configuration (Docker Paths) ---
# نقرأ البيانات اللي الموديل والـ ETL جهزوها في الخطوات اللي فاتت
DATA_PATH = '/data/final_data.csv'
METADATA_PATH = '/data/hotel_data_processed.joblib'

# Global Variables to hold data in memory
df_hotels = None
hotel_features = None
global_max_price = 5000
available_amenities_list = []

def load_system():
    global df_hotels, global_max_price, available_amenities_list
    
    print("⏳ Loading Hotel System...")
    
    # 1. Load Main Data
    if os.path.exists(DATA_PATH):
        try:
            df_hotels = pd.read_csv(DATA_PATH)
            
            # Clean & Fix Types
            df_hotels['Latitude'] = pd.to_numeric(df_hotels['Latitude'], errors='coerce')
            df_hotels['Longitude'] = pd.to_numeric(df_hotels['Longitude'], errors='coerce')
            df_hotels['ai_score'] = pd.to_numeric(df_hotels['ai_score'], errors='coerce').fillna(0.0)
            df_hotels['price'] = pd.to_numeric(df_hotels['price'], errors='coerce').fillna(0.0)
            
            # Set Max Price for UI Slider
            global_max_price = int(df_hotels['price'].max()) + 50
            
            print(f"✅ Data Loaded: {len(df_hotels)} hotels.")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            df_hotels = pd.DataFrame()
    else:
        print(f"⚠️ Warning: {DATA_PATH} not found. Make sure ETL finished.")
        df_hotels = pd.DataFrame()

    # 2. Extract Amenities for UI Filter
    # بنحاول نطلع أسماء الخدمات من أعمدة الداتا (اللي هي 0 و 1)
    ignore_cols = ['name', 'location', 'price', 'rating', 'Latitude', 'Longitude', 
                   'images', 'gps_link', 'id', 'sentiment_score', 'ai_score', 
                   'image1', 'image2', 'image3', 'image4', 'image5', 'amenities', 'OpenCage Note', 'price_clean']
    
    if df_hotels is not None and not df_hotels.empty:
        available_amenities_list = [col for col in df_hotels.columns 
                                    if col not in ignore_cols and df_hotels[col].nunique() <= 3]
        print(f"✅ Loaded {len(available_amenities_list)} amenities for filtering.")

# Initialize App
with app.app_context():
    load_system()

# --- Helper Functions ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in KM between two coordinates"""
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]): return 99999
    R = 6371 # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def format_hotel_response(row, distance=0):
    """Convert DataFrame row to JSON format for UI"""
    # Handle Images
    images = []
    for i in range(1, 6):
        img = row.get(f'image{i}')
        if pd.notna(img) and str(img).startswith('http'):
            images.append(str(img))
    
    main_image = images[0] if images else "https://placehold.co/600x400?text=No+Image"

    return {
        'id': int(row.name), # Use Index as ID
        'name': str(row['name']),
        'price': float(row['price']),
        'rating': float(row['rating']) if pd.notna(row['rating']) else 0.0,
        'ai_score': round(float(row.get('ai_score', 0)), 1), # The Predicted Score!
        'lat': float(row['Latitude']),
        'lng': float(row['Longitude']),
        'image_url': main_image,
        'all_images': images,
        'distance_km': round(distance, 1),
        'location': str(row.get('location', ''))
    }

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_panel')
def search_panel_route():
    return render_template('search_panel.html')

@app.route('/hotel/<int:hotel_id>')
def hotel_details(hotel_id):
    if df_hotels is not None and 0 <= hotel_id < len(df_hotels):
        row = df_hotels.iloc[hotel_id]
        hotel_data = format_hotel_response(row)
        # Get amenities explicitly
        active_amenities = []
        for feature in available_amenities_list:
            if row.get(feature, 0) == 1:
                active_amenities.append(feature.replace('_', ' ').title())
        
        hotel_data['amenities_list'] = active_amenities
        return render_template('details.html', hotel=hotel_data)
    return "Hotel not found", 404

@app.route('/api/metadata')
def get_metadata():
    return jsonify({
        'max_price': global_max_price, 
        'amenities': available_amenities_list # Return top 20 to avoid clutter
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_lat = float(data.get('lat'))
        user_lng = float(data.get('lng'))
        radius = float(data.get('radius', 50))
        max_price = float(data.get('max_price', global_max_price))
        
        if df_hotels is None or df_hotels.empty:
            return jsonify({'status': 'error', 'message': 'System not ready'})

        # 1. Filter by Price (Fastest)
        candidates = df_hotels[df_hotels['price'] <= max_price].copy()
        
        # 2. Calculate Distance (Vectorized if possible, but loop is safe for <5000 rows)
        # To speed up, we can first filter by rough box coordinates
        candidates['distance_km'] = candidates.apply(
            lambda row: haversine(user_lat, user_lng, row['Latitude'], row['Longitude']), axis=1
        )
        
        # 3. Filter by Radius
        nearby = candidates[candidates['distance_km'] <= radius].copy()
        
        if nearby.empty:
            # Fallback: Get closest 5 if nothing in radius
            nearby = candidates.sort_values('distance_km').head(5).copy()
            message = "No hotels in range, showing closest options."
        else:
            message = f"Found {len(nearby)} hotels nearby."

        # 4. Ranking Logic (The Core AI Value) 🧠
        # Score = (AI_Quality * 0.7) + (Distance_Factor * 0.3)
        # Closer hotels get higher distance score
        max_dist = nearby['distance_km'].max() + 0.1
        nearby['dist_score'] = 1 - (nearby['distance_km'] / max_dist)
        
        # Normalizing AI Score (0-5) -> (0-1)
        nearby['quality_score'] = nearby['ai_score'] / 5.0
        
        nearby['final_rank'] = (nearby['quality_score'] * 0.7) + (nearby['dist_score'] * 0.3)
        
        # Sort
        top_hotels = nearby.sort_values('final_rank', ascending=False).head(20)
        
        response_data = [format_hotel_response(row, row['distance_km']) for _, row in top_hotels.iterrows()]

        return jsonify({'status': 'success', 'message': message, 'data': response_data})

    except Exception as e:
        print(f"Recommendation Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
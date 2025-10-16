from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
import json
import os
import random
import time
from PIL import Image
import io
import base64

# Attempt to import numpy, but use conditional import since it might not be installed
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

app = Flask(__name__)
# IMPORTANT: Use a complex secret key in production!
app.secret_key = "supersecretkey"

# Backendless API details
BACKENDLESS_APP_ID = "AE6D2608-FF11-42EB-90DC-A30443D16A35"
BACKENDLESS_REST_API_KEY = "8A2D3DEC-606A-487A-9231-33FB936C713E"
BASE_URL = f"https://api.backendless.com/{BACKENDLESS_APP_ID}/{BACKENDLESS_REST_API_KEY}"

# Endpoint for User table operations
USER_TABLE_URL = f"{BASE_URL}/data/Users"
USER_API_URL = f"{BASE_URL}/users"

# Global variable to hold product embeddings
PRODUCT_EMBEDDINGS = None

# --- Product Data and Utility Functions ---

def load_products():
    """
    Loads product data (metadata) from a JSON file and tries to load 
    fashion embeddings from the NPY file.
    """
    global PRODUCT_EMBEDDINGS

    # --- 1. Load Product Metadata (Required for front-end) ---
    products_metadata = []
    
    # HARDCODED FALLBACK LIST (Ensures app runs even without data.json)
    default_products = [
        {"id": 1, "name": "Classic White Tee", "category": "T-shirt", "price": 500, "imageUrl": "https://placehold.co/300x400/CCCCCC/333333?text=Tee", "color": "White", "material": "Cotton", "style": "Sporty, Casual", "fit": "Regular", "gender": "Unisex", "combo_type": "Top"},
        {"id": 2, "name": "Slim Fit Jeans", "category": "Jeans", "price": 1800, "imageUrl": "https://placehold.co/300x400/556B2F/FFFFFF?text=Jeans", "color": "Dark Blue", "material": "Denim", "style": "Stylish, Edgy", "fit": "Slim", "gender": "Male", "combo_type": "Bottom"},
        {"id": 3, "name": "Red Sneakers", "category": "Sneakers", "price": 3500, "imageUrl": "https://placehold.co/300x400/DC143C/FFFFFF?text=Shoes", "color": "Red", "material": "Leather", "style": "Sporty, Edgy", "fit": "Standard", "gender": "Unisex", "combo_type": "Shoes"},
        {"id": 4, "name": "Minimalist Watch", "category": "Watch", "price": 4200, "imageUrl": "https://placehold.co/300x400/000000/FFFFFF?text=Watch", "color": "Black", "material": "Metal", "style": "Decent, Stylish", "fit": "NA", "gender": "Unisex", "combo_type": "Accessory"},
        {"id": 5, "name": "Summer Floral Dress", "category": "Dress", "price": 2500, "imageUrl": "https://placehold.co/300x400/FFB6C1/000000?text=Dress", "color": "Pink", "material": "Rayon", "style": "Stylish, Decent", "fit": "Flowy", "gender": "Female", "combo_type": "Top"},
    ]

    try:
        if os.path.exists('data.json'):
            with open('data.json', 'r') as f:
                products_metadata = json.load(f)
        else:
            print("Warning: 'data.json' not found. Using hardcoded product list.")
            products_metadata = default_products
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from data.json. Using hardcoded list.")
        products_metadata = default_products

    # --- 2. Load Fashion Embeddings (for advanced logic) ---
    if PRODUCT_EMBEDDINGS is None and NUMPY_AVAILABLE:
        try:
            # Assuming fashion_embeddings.npy exists and matches product indices
            if os.path.exists('fashion_embeddings.npy'):
                PRODUCT_EMBEDDINGS = np.load('fashion_embeddings.npy')
                print(f"Loaded {PRODUCT_EMBEDDINGS.shape[0]} embeddings from fashion_embeddings.npy.")
            else:
                # SIMULATION: Create dummy embeddings if the file is missing
                num_products = len(products_metadata)
                if num_products > 0:
                    PRODUCT_EMBEDDINGS = np.random.rand(num_products, 128) # 128-dim embedding
                    print("Simulating fashion_embeddings.npy for recommendation logic.")
        except Exception as e:
            print(f"Error loading fashion_embeddings.npy: {e}")
            PRODUCT_EMBEDDINGS = None

    return products_metadata

def get_user_profile(token, object_id, max_retries=3):
    """
    Fetches the current user's profile details from Backendless,
    implementing exponential backoff for Connection Errors.
    """
    if not token or not object_id:
        return None

    headers = {"user-token": token}

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{USER_TABLE_URL}/{object_id}", headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                if attempt == 0:
                    print(f"Backendless HTTP Error {response.status_code} for user {object_id}: {response.text}")
                return None

        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Connection error (Attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Connection error after {max_retries} attempts. Giving up.")
                return None
        except Exception as e:
            print(f"Unexpected error during profile fetch: {e}")
            return None

    return None

# --- Recommendation Logic ---

COMBO_CATEGORIES = ['Top', 'Bottom', 'Shoes', 'Accessory']

def generate_combos(all_products, purpose, look_type, user_profile=None):
    """Generates 3 dummy combos based on user input, style, and gender from profile."""
    random.seed(purpose + look_type)

    # NOTE: In a production environment, this function would leverage PRODUCT_EMBEDDINGS
    # to find products that are semantically close or match a vector generated from 
    # the user profile/intent, greatly improving recommendation quality.
    # e.g., using numpy.linalg.norm(PRODUCT_EMBEDDINGS - user_vector, axis=1)

    # 1. Determine Gender Preference
    user_gender = user_profile.get('gender', '').lower() if user_profile else ''

    # 2. Base Product Filtering (by Gender)
    gender_filtered_products = []
    if user_gender and user_gender != 'other':
        for p in all_products:
            product_gender = p.get('gender', 'unisex').lower()
            # Match user gender, or accept unisex
            if product_gender == user_gender or product_gender == 'unisex':
                 gender_filtered_products.append(p)
    else:
        # Fallback to all products if gender is missing or 'other'
        gender_filtered_products = all_products

    # 3. Style Filtering
    # Filter products whose style contains the look_type keyword, from the gender-filtered list
    style_filtered_products = [
        p for p in gender_filtered_products
        if look_type.lower() in p.get('style', 'casual').lower()
    ] or gender_filtered_products # Fallback to gender_filtered_products if no style match

    combos = []

    for i in range(3): # Generate 3 Combos
        combo_list = []

        # Helper function to pick items with priority on filtered lists
        def pick_item(ctype, exclude_ids):
            # 1st priority: style + gender filtered
            candidates = [p for p in style_filtered_products if p.get('combo_type') == ctype and p['id'] not in exclude_ids]
            if not candidates:
                # 2nd priority: just gender filtered
                candidates = [p for p in gender_filtered_products if p.get('combo_type') == ctype and p['id'] not in exclude_ids]
            if not candidates:
                # 3rd priority: all products (last resort)
                candidates = [p for p in all_products if p.get('combo_type') == ctype and p['id'] not in exclude_ids]

            return random.choice(candidates) if candidates else None

        used_ids = set()

        # Select items for the combo
        top = pick_item('Top', used_ids)
        if top: used_ids.add(top['id'])

        bottom = pick_item('Bottom', used_ids)
        if bottom: used_ids.add(bottom['id'])

        shoes = pick_item('Shoes', used_ids)
        if shoes: used_ids.add(shoes['id'])

        accessory = pick_item('Accessory', used_ids)
        if accessory: used_ids.add(accessory['id'])

        combo_list = [item for item in [top, bottom, shoes, accessory] if item]

        if len(combo_list) == len(COMBO_CATEGORIES):
            combos.append({
                'id': i + 1,
                'products': combo_list,
                'reason': f"This combination utilizes **{top['material']}** and complementary **{bottom['color']}** to achieve a distinct **{look_type}** look suitable for your **{purpose}** event. The accessories add a perfect final touch to match your profile."
            })

    return combos

# --- Flask Routes ---

@app.route("/")
def home():
    """Landing page for login/registration."""
    return render_template("main.html")


@app.route("/register", methods=["POST"])
def register():
    data = {
        "email": request.form["email"],
        "password": request.form["password"],
        "name": request.form["name"]
    }
    response = requests.post(USER_API_URL + "/register", json=data)
    if response.status_code == 200:
        flash("Account created successfully! Please log in.", 'info')
        return redirect(url_for("home"))
    else:
        try:
            error_message = response.json().get('message', 'Registration failed! Try again.')
        except:
            error_message = 'Registration failed! Try again.'
        flash(error_message, 'error')
        return redirect(url_for("home"))


@app.route("/login", methods=["POST"])
def login():
    data = {
        "login": request.form["email"],
        "password": request.form["password"]
    }
    response = requests.post(USER_API_URL + "/login", json=data)
    if response.status_code == 200:
        user = response.json()
        session["user_email"] = user.get("email")
        session["user_name"] = user.get("name", user.get("email").split('@')[0])
        session["user_object_id"] = user.get("objectId")
        session["user_token"] = user.get("user-token")
        return redirect(url_for("index"))
    else:
        flash("Invalid credentials!", 'error')
        return redirect(url_for("home"))


@app.route("/logout")
def logout():
    """Logs out the user and clears the session."""
    if "user_token" in session:
        headers = {"user-token": session["user_token"]}
        # Attempt to call Backendless logout endpoint
        requests.get(USER_API_URL + "/logout", headers=headers)

    session.pop("user_email", None)
    session.pop("user_name", None)
    session.pop("user_object_id", None)
    session.pop("user_token", None)
    flash("You have logged out.", 'info')
    return redirect(url_for("home"))


@app.route("/index")
def index():
    """Main page showing products and handling profile/recommendation data."""
    if "user_email" not in session:
        flash("You must be logged in to view this page.", 'warning')
        return redirect(url_for("home"))

    all_products = load_products()
    search_query = request.args.get("search", "").lower()
    
    # Profile/View Logic
    user_profile = get_user_profile(session.get("user_token"), session.get("user_object_id"))
    view_mode = request.args.get("view", "products")

    products_for_grid = all_products # Start with the full list
    single_product = None # Variable to hold the specific product object

    # Handle Product Grid Filtering
    if view_mode == 'products' and search_query:
        # Filter only if viewing the main product grid
        products_for_grid = [p for p in all_products if search_query in p['name'].lower() or search_query in p['category'].lower()]
        if not products_for_grid:
             flash(f"No results found for '{search_query}'. Showing all products.", 'info')
             products_for_grid = all_products
    
    # Handle Single Product Lookup (FIX)
    if view_mode == 'single_product':
        requested_product_id = request.args.get('product_id', type=int)
        if requested_product_id:
            # Look up product in the *full* list using next()
            single_product = next((p for p in all_products if p.get('id') == requested_product_id), None)
            if not single_product:
                # If product not found, change view_mode back to products and flash an error
                flash(f"Error: Product ID {requested_product_id} not found.", 'error')
                view_mode = 'products' 

    # Recommendation Intent logic
    purpose = request.args.get('purpose')
    look_type = request.args.get('look_type')

    combos = None
    if view_mode == 'recommendations' and purpose and look_type:
        # Pass user_profile for gender-based filtering
        combos = generate_combos(all_products, purpose, look_type, user_profile)

    # Train Room Logic
    selected_combo_id = request.args.get('combo_id')
    train_room_items = None
    combo_reason = None

    if view_mode == 'train_room' and selected_combo_id:
        if not (purpose and look_type):
             # Safety redirect if params are missing for combo generation
             flash("Missing recommendation parameters.", 'error')
             return redirect(url_for('index', view='intent'))

        # Pass user_profile to ensure consistency with the recommendation view
        all_combos = generate_combos(all_products, purpose, look_type, user_profile)
        selected_combo = next((c for c in all_combos if str(c['id']) == selected_combo_id), None)

        if selected_combo:
            train_room_items = selected_combo['products']
            combo_reason = selected_combo['reason']
        else:
            flash("Selected combo not found.", 'error')
            return redirect(url_for('index', view='recommendations', purpose=purpose, look_type=look_type))

    return render_template("index.html",
        user=session["user_name"],
        products=products_for_grid, # Used for the main grid
        single_product=single_product, # NEW: Used for single product view
        search_query=search_query,
        user_profile=user_profile,
        view_mode=view_mode,
        combos=combos,
        purpose=purpose,
        look_type=look_type,
        train_room_items=train_room_items,
        combo_reason=combo_reason
    )


@app.route("/save_profile", methods=["POST"])
def save_profile():
    if "user_object_id" not in session or "user_token" not in session:
        flash("Session expired. Please log in again.", 'error')
        return redirect(url_for("home"))

    try:
        profile_data = {
            "name": request.form.get("name"),
            # Ensure proper type casting for numerical fields
            "age": int(request.form.get("age") or 0),
            "gender": request.form.get("gender"),
            "height": float(request.form.get("height") or 0.0),
            "weight": float(request.form.get("weight") or 0.0),
            "bodyShape": request.form.get("bodyShape"),
            "skinTone": request.form.get("skinTone")
        }
    except (ValueError, TypeError):
        flash("Invalid input for Age, Height, or Weight. Please check your data.", 'error')
        return redirect(url_for("index", view="profile", edit="true"))

    object_id = session["user_object_id"]
    token = session["user_token"]

    headers = {
        "user-token": token,
        "Content-Type": "application/json"
    }

    # Using retry mechanism for profile updates
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.put(f"{USER_TABLE_URL}/{object_id}", headers=headers, json=profile_data)

            if response.status_code == 200:
                flash("Profile saved successfully! 🎉", 'info')
                session["user_name"] = profile_data["name"]
                return redirect(url_for("index", view="profile"))
            else:
                flash("Failed to save profile. Check Backendless permissions and try again.", 'error')
                print(f"Backendless Update Error: {response.status_code} - {response.text}")
                return redirect(url_for("index", view="profile", edit="true"))

        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Connection error (Attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                flash("Failed to connect to the server. Please check your network.", 'error')
                return redirect(url_for("index", view="profile", edit="true"))
        except Exception as e:
            print(f"Unexpected error during profile save: {e}")
            flash("An unexpected error occurred while saving your profile.", 'error')
            return redirect(url_for("index", view="profile", edit="true"))

    # Fallback return
    return redirect(url_for("index", view="profile"))


# --- AI Skin Tone Detection Route ---
@app.route("/detect_skin_tone", methods=["POST"])
def detect_skin_tone():
    """
    Receives a base64 encoded image, simulates skin tone detection, and
    returns the closest matching tone from the predefined list.
    """
    if "user_object_id" not in session:
        return {"status": "error", "message": "User not logged in"}, 401

    try:
        data = request.json
        img_data = data["image"].split(',')[1] # Remove the 'data:image/jpeg;base64,' prefix
        image_bytes = base64.b64decode(img_data)
        
        # Open the image using PIL (Pillow)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Predefined skin tones (Name, RGB Tuple)
        tones = [
            ('Porcelain', (247, 231, 216)), 
            ('Light Beige', (232, 199, 168)), 
            ('Warm Tan', (215, 167, 121)), 
            ('Olive', (160, 118, 77)), 
            ('Deep Brown', (77, 47, 40))
        ]
        
        # SIMULATION: Use the average color of a central patch and match it to the closest predefined tone.
        
        # 1. Get average RGB of a central 50x50 patch (simulated skin area)
        width, height = img.size
        left = width // 2 - 25
        top = height // 2 - 25
        right = width // 2 + 25
        bottom = height // 2 + 25
        
        center_patch = img.crop((left, top, right, bottom))
        # Handle grayscale/single-channel images by converting to RGB
        if center_patch.mode != 'RGB':
            center_patch = center_patch.convert('RGB')
            
        avg_rgb = [int(sum(c) / len(c)) for c in zip(*center_patch.getdata())][:3] # Get average R, G, B
        
        # 2. Find the closest predefined tone
        closest_tone = None
        min_distance = float('inf')

        for name, rgb in tones:
            r1, g1, b1 = avg_rgb
            r2, g2, b2 = rgb
            # Euclidean distance in RGB space
            distance = ((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_tone = (name, rgb)
                
        detected_tone, detected_rgb = closest_tone
        detected_hex = '#%02x%02x%02x' % detected_rgb
        
        return {
            "status": "success",
            "skinTone": detected_tone,
            "hex": detected_hex,
            "message": f"Detected average color ({avg_rgb}) matched to {detected_tone}."
        }

    except Exception as e:
        print(f"Error in skin tone detection: {e}")
        return {"status": "error", "message": "Server-side processing failed"}, 500


if __name__ == "__main__":
    # Load products (and embeddings if available) when the app starts
    load_products() 
    app.run(debug=True)

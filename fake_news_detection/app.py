# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from models import db, User # Import db and User from models.py
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Import VADER for sentiment analysis

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')  # Add this too for sentiment analysis





app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_new_key_123' # Keep this secret and strong in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress SQLAlchemy warning

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# ---------------------------
# Load vectorizer and model
# ---------------------------
vectorizer_path = 'tfidf_vectorizer.pkl'
model_path = 'fake_news_model.pkl'

def load_model_and_vectorizer():
    """
    Loads the pre-trained TF-IDF vectorizer and classification model.
    Prints a message if files are not found.
    """
    vectorizer = None
    model = None
    if not os.path.exists(vectorizer_path):
        print(f"❌ TF-IDF vectorizer not found at '{vectorizer_path}'. Please run train_model.py first.")
    if not os.path.exists(model_path):
        print(f"❌ Classification model not found at '{model_path}'. Please run train_model.py first.")
    
    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        try:
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
            print("✅ Model and vectorizer loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model or vectorizer: {e}")
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

@login_manager.user_loader
def load_user(user_id):
    """
    Callback function for Flask-Login to load a user from the database.
    """
    return User.query.get(int(user_id))

@app.before_request
def create_tables():
    """
    Ensures that database tables are created and a default admin user exists
    before the first request is handled.
    """
    # This check is important to prevent re-creation on every request
    # and only runs if the database file doesn't exist.
    if not os.path.exists('users.db'):
        with app.app_context():
            db.create_all()
            # Create a default admin user if one doesn't exist
            if not User.query.filter_by(username='admin').first():
                admin_user = User(username='admin')
                admin_user.set_password('adminpass')
                db.session.add(admin_user)
                db.session.commit()
                print("🛡️ Default admin user created (username: admin, password: adminpass)")

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    """
    Redirects authenticated users to the prediction page,
    otherwise redirects to the login page.
    """
    return redirect(url_for('predict_news')) if current_user.is_authenticated else redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handles user login.
    If already logged in, redirects to predict_news.
    """
    if current_user.is_authenticated:
        return redirect(url_for('predict_news'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash('✅ Logged in successfully!', 'success')
            return redirect(url_for('predict_news'))
        flash('❌ Invalid username or password.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handles new user registration.
    If already logged in, redirects to predict_news.
    """
    if current_user.is_authenticated:
        return redirect(url_for('predict_news'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('⚠️ Username already exists. Please choose a different one.', 'danger')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('✅ Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """
    Logs out the current user.
    """
    logout_user()
    flash('ℹ️ You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict_news():
    """
    Handles news text prediction (fake or real) and sentiment analysis.
    Requires user to be logged in.
    """
    prediction_result = None
    confidence_score = None
    predicted_label = None
    news_text = ""
    sentiment_scores = None # Initialize sentiment scores

    if request.method == 'POST':
        news_text = request.form['news_text']
        if not news_text.strip():
            flash("⚠️ Please enter some news text to analyze.", "warning")
        elif not vectorizer or not model:
            flash("⚠️ NLP model or vectorizer not loaded. Please ensure 'train_model.py' was run successfully.", "danger")
        else:
            # Perform Fake News Prediction
            processed_text = preprocess_text(news_text)
            print(f"🧹 Cleaned Text for Prediction: '{processed_text}'")

            input_vec = vectorizer.transform([processed_text])
            prediction = model.predict(input_vec)[0]
            prediction_proba = model.predict_proba(input_vec)[0]

            print(f"🔮 Raw Prediction: {prediction}")
            print(f"📊 Probabilities: Fake={prediction_proba[0]:.4f}, Real={prediction_proba[1]:.4f}")

            if prediction == 0:
                predicted_label = "FAKE"
                confidence_score = prediction_proba[0] * 100
                prediction_result = f"🟥 FAKE NEWS (Confidence: {confidence_score:.2f}%)"
            else:
                predicted_label = "REAL"
                confidence_score = prediction_proba[1] * 100
                prediction_result = f"🟩 REAL NEWS (Confidence: {confidence_score:.2f}%)"

            if confidence_score < 70:
                flash(f"⚠️ Warning: Model confidence is low ({confidence_score:.2f}%). This prediction might be less reliable.", "warning")

            # Perform Sentiment Analysis
            sentiment_scores = analyzer.polarity_scores(news_text)
            print(f"😊 Sentiment Scores: {sentiment_scores}")

    return render_template('predict.html',
                           prediction_result=prediction_result,
                           news_text=news_text,
                           predicted_label=predicted_label,
                           confidence_score=confidence_score,
                           sentiment_scores=sentiment_scores) # Pass sentiment scores to template
    
# ---------------------------
# Preprocessing Function (Same as in train_model.py for consistency)
# ---------------------------
def preprocess_text(text):
    """
    Cleans and preprocesses text for NLP tasks.
    Steps include:
    - Lowercasing
    - Removing URLs
    - Removing HTML tags
    - Removing non-alphabetic characters and numbers
    - Tokenization
    - Stop word removal
    - Stemming
    """
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-alphabetic characters
    text = re.sub(r'\d+', '', text) # Remove numbers
    
    tokens = nltk.word_tokenize(text) # Tokenize text
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Filter out stop words and apply stemming
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)

# ---------------------------
# Run the App
# ---------------------------
if __name__ == '__main__':
    # Ensure tables are created when the app starts
    with app.app_context():
        db.create_all() # This will create tables if they don't exist
        # Also ensure the admin user is created if not present
        if not User.query.filter_by(username='admin').first():
            admin_user = User(username='admin')
            admin_user.set_password('adminpass')
            db.session.add(admin_user)
            db.session.commit()
            print("🛡️ Default admin user created (username: admin, password: adminpass)")
            
    app.run(debug=True) # debug=True enables auto-reloading and helpful error messages

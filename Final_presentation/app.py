"""
Streamlit Dashboard for ML Models
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.set_page_config(page_title="ML Model Dashboard", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 { color: #fff !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Machine Learning Model Dashboard")
st.markdown("### Interactive predictions using trained deep learning models")

# Check for saved models
SEATTLE_DIR = 'Seattle Weather Prediction Dataset/saved_models'
CLIMATE_DIR = '4_Daily Climate Time Series Data/saved_models'
CORGIS_DIR = '3_CORGIS_Weather_TimeSeries/saved_models'
TWITTER_DIR = '2_Twitter Sentiment Analysis/saved_models'
TESS_DIR = '1_Toronto Emotional Speech Set (TESS)/saved_models'

DIRS = [SEATTLE_DIR, CLIMATE_DIR, CORGIS_DIR, TWITTER_DIR, TESS_DIR]
if not any(os.path.exists(d) for d in DIRS):
    st.error("⚠️ No saved models found! Please run the notebooks first to train and save models.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌧️ Seattle Weather", "🌡️ Daily Climate", "🌤️ CORGIS Weather",
    "💬 Twitter Sentiment", "🎤 TESS Emotion"
])

# ==================== TAB 1: Seattle Weather ====================
with tab1:
    st.header("🌧️ Seattle Weather Prediction")
    st.markdown("Predict weather type based on climate features")
    
    try:
        MODEL_DIR = SEATTLE_DIR
        from tensorflow.keras.models import load_model
        
        scaler = pickle.load(open(f'{MODEL_DIR}/seattle_scaler.pkl', 'rb'))
        le = pickle.load(open(f'{MODEL_DIR}/seattle_label_encoder.pkl', 'rb'))
        
        model_choice = st.selectbox("Select Model", ["ANN", "RNN", "LSTM", "GRU"], key="seattle_model")
        model = load_model(f'{MODEL_DIR}/seattle_{model_choice.lower()}_model.keras')
        
        col1, col2 = st.columns(2)
        with col1:
            precip = st.slider("Precipitation (mm)", 0.0, 60.0, 5.0, key="s_precip")
            temp_max = st.slider("Max Temperature (C)", -5.0, 40.0, 15.0, key="s_tmax")
        with col2:
            temp_min = st.slider("Min Temperature (C)", -10.0, 25.0, 8.0, key="s_tmin")
            wind = st.slider("Wind Speed (m/s)", 0.0, 10.0, 3.0, key="s_wind")
        
        if st.button("Predict Weather", key="seattle_btn"):
            features = scaler.transform([[precip, temp_max, temp_min, wind]])
            if model_choice == "ANN":
                pred = model.predict(features)
            else:
                pred = model.predict(features.reshape(1, 1, -1))
            
            predicted_class = le.inverse_transform([np.argmax(pred)])[0]
            confidence = np.max(pred) * 100
            
            st.success(f"**Predicted Weather: {predicted_class.upper()}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            st.bar_chart(pd.DataFrame(pred[0], index=le.classes_, columns=["Probability"]))
        
        # Show comparison results
        if os.path.exists(f'{MODEL_DIR}/seattle_comparison_results.csv'):
            st.subheader("Model Comparison")
            comp = pd.read_csv(f'{MODEL_DIR}/seattle_comparison_results.csv', index_col=0)
            st.dataframe(comp.style.highlight_max(axis=0), use_container_width=True)
    except Exception as e:
        st.warning(f"Seattle models not loaded. Run the notebook first. Error: {e}")

# ==================== TAB 2: Daily Climate ====================
with tab2:
    st.header("🌡️ Daily Climate Forecasting")
    st.markdown("Forecast Delhi mean temperature")
    
    try:
        MODEL_DIR = CLIMATE_DIR
        from tensorflow.keras.models import load_model
        
        scaler = pickle.load(open(f'{MODEL_DIR}/climate_scaler.pkl', 'rb'))
        config = pickle.load(open(f'{MODEL_DIR}/climate_config.pkl', 'rb'))
        
        model_choice = st.selectbox("Select Model", ["ANN", "RNN", "LSTM", "GRU"], key="climate_model")
        model = load_model(f'{MODEL_DIR}/climate_{model_choice.lower()}_model.keras')
        
        st.markdown("**Enter last 30 days of data (or use defaults):**")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Current Avg Temp (C)", value=25.0, key="c_temp")
            humidity = st.number_input("Current Humidity", value=60.0, key="c_hum")
        with col2:
            wind = st.number_input("Wind Speed", value=7.0, key="c_wind")
            pressure = st.number_input("Mean Pressure", value=1015.0, key="c_press")
        
        if st.button("Forecast Temperature", key="climate_btn"):
            # Create synthetic sequence using input as base
            window = config['window_size']
            base = np.array([[temp, humidity, wind, pressure]])
            sequence = np.tile(base, (window, 1))
            # Add small noise for realism
            noise = np.random.normal(0, 0.5, sequence.shape)
            sequence = sequence + noise
            
            scaled_seq = scaler.transform(sequence)
            
            if model_choice == "ANN":
                X_input = scaled_seq.reshape(1, -1)
            else:
                X_input = scaled_seq.reshape(1, window, len(config['feature_cols']))
            
            pred_scaled = model.predict(X_input)[0, 0]
            
            # Inverse transform
            dummy = np.zeros((1, len(config['feature_cols'])))
            dummy[0, config['target_idx']] = pred_scaled
            pred_temp = scaler.inverse_transform(dummy)[0, config['target_idx']]
            
            st.success(f"**Predicted Mean Temperature: {pred_temp:.1f} C**")
        
        if os.path.exists(f'{MODEL_DIR}/climate_comparison_results.csv'):
            st.subheader("Model Comparison")
            comp = pd.read_csv(f'{MODEL_DIR}/climate_comparison_results.csv', index_col=0)
            st.dataframe(comp.style.highlight_min(subset=['RMSE', 'MAE'], axis=0).highlight_max(subset=['R²'], axis=0), use_container_width=True)
    except Exception as e:
        st.warning(f"Climate models not loaded. Error: {e}")

# ==================== TAB 3: CORGIS Weather ====================
with tab3:
    st.header("🌤️ CORGIS Weather Forecasting")
    st.markdown("Forecast temperature for selected city")
    
    try:
        MODEL_DIR = CORGIS_DIR
        from tensorflow.keras.models import load_model
        
        scaler = pickle.load(open(f'{MODEL_DIR}/corgis_scaler.pkl', 'rb'))
        config = pickle.load(open(f'{MODEL_DIR}/corgis_config.pkl', 'rb'))
        
        model_choice = st.selectbox("Select Model", ["ANN", "RNN", "LSTM", "GRU"], key="corgis_model")
        model = load_model(f'{MODEL_DIR}/corgis_{model_choice.lower()}_model.keras')
        
        col1, col2 = st.columns(2)
        with col1:
            avg_temp = st.number_input("Avg Temp", value=55.0, key="co_avg")
            max_temp = st.number_input("Max Temp", value=65.0, key="co_max")
            min_temp = st.number_input("Min Temp", value=45.0, key="co_min")
        with col2:
            wind_speed = st.number_input("Wind Speed", value=10.0, key="co_wind")
            precip = st.number_input("Precipitation", value=0.5, key="co_precip")
        
        if st.button("Forecast", key="corgis_btn"):
            window = config['window_size']
            base = np.array([[avg_temp, max_temp, min_temp, wind_speed, precip]])
            sequence = np.tile(base, (window, 1)) + np.random.normal(0, 1, (window, 5))
            scaled_seq = scaler.transform(sequence)
            
            if model_choice == "ANN":
                X_input = scaled_seq.reshape(1, -1)
            else:
                X_input = scaled_seq.reshape(1, window, len(config['feature_cols']))
            
            pred_scaled = model.predict(X_input)[0, 0]
            dummy = np.zeros((1, len(config['feature_cols'])))
            dummy[0, config['target_idx']] = pred_scaled
            pred_val = scaler.inverse_transform(dummy)[0, config['target_idx']]
            
            st.success(f"**Predicted Avg Temperature: {pred_val:.1f}**")
        
        if os.path.exists(f'{MODEL_DIR}/corgis_comparison_results.csv'):
            st.subheader("Model Comparison")
            comp = pd.read_csv(f'{MODEL_DIR}/corgis_comparison_results.csv', index_col=0)
            st.dataframe(comp, use_container_width=True)
    except Exception as e:
        st.warning(f"CORGIS models not loaded. Error: {e}")

# ==================== TAB 4: Twitter Sentiment ====================
with tab4:
    st.header("💬 Twitter Sentiment Analysis")
    st.markdown("Analyze the sentiment of any text")
    
    try:
        MODEL_DIR = TWITTER_DIR
        import re
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        tokenizer = pickle.load(open(f'{MODEL_DIR}/twitter_tokenizer.pkl', 'rb'))
        le = pickle.load(open(f'{MODEL_DIR}/twitter_label_encoder.pkl', 'rb'))
        config = pickle.load(open(f'{MODEL_DIR}/twitter_config.pkl', 'rb'))
        
        model_choice = st.selectbox("Select Model", ["RNN", "LSTM", "GRU"], key="twitter_model")
        model = load_model(f'{MODEL_DIR}/twitter_{model_choice.lower()}_model.keras')
        
        text_input = st.text_area("Enter tweet or text:", 
                                   "I absolutely love this product! Best purchase ever!",
                                   key="tweet_input")
        
        if st.button("Analyze Sentiment", key="twitter_btn"):
            # Clean text
            text = str(text_input).lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=config['max_len'], padding='post', truncating='post')
            
            pred = model.predict(padded)
            sentiment = le.inverse_transform([np.argmax(pred)])[0]
            confidence = np.max(pred) * 100
            
            emoji_map = {'Positive': '😊', 'Negative': '😞', 'Neutral': '😐', 'Irrelevant': '🤷'}
            emoji = emoji_map.get(sentiment, '🤔')
            
            st.success(f"**Sentiment: {emoji} {sentiment}** (Confidence: {confidence:.1f}%)")
            
            st.bar_chart(pd.DataFrame(pred[0], index=le.classes_, columns=["Probability"]))
        
        if os.path.exists(f'{MODEL_DIR}/twitter_comparison_results.csv'):
            st.subheader("Model Comparison")
            comp = pd.read_csv(f'{MODEL_DIR}/twitter_comparison_results.csv', index_col=0)
            st.dataframe(comp, use_container_width=True)
    except Exception as e:
        st.warning(f"Twitter models not loaded. Error: {e}")

# ==================== TAB 5: TESS Emotion ====================
with tab5:
    st.header("🎤 TESS Emotion Classification")
    st.markdown("Upload a WAV audio file to classify emotion")
    
    try:
        MODEL_DIR = TESS_DIR
        from tensorflow.keras.models import load_model
        import librosa
        
        scaler = pickle.load(open(f'{MODEL_DIR}/tess_scaler.pkl', 'rb'))
        le = pickle.load(open(f'{MODEL_DIR}/tess_label_encoder.pkl', 'rb'))
        config = pickle.load(open(f'{MODEL_DIR}/tess_config.pkl', 'rb'))
        
        model_choice = st.selectbox("Select Model", ["RNN", "LSTM", "GRU"], key="tess_model")
        model = load_model(f'{MODEL_DIR}/tess_{model_choice.lower()}_model.keras')
        
        uploaded_file = st.file_uploader("Upload WAV file", type=['wav'], key="tess_upload")
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Classify Emotion", key="tess_btn"):
                # Save temp file and process
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                audio, sr = librosa.load(tmp_path, duration=3, sr=22050)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config['n_mfcc']).T
                
                if mfcc.shape[0] < config['max_len']:
                    mfcc = np.pad(mfcc, ((0, config['max_len'] - mfcc.shape[0]), (0, 0)))
                else:
                    mfcc = mfcc[:config['max_len']]
                
                # Scale
                mfcc_flat = mfcc.reshape(-1, config['n_mfcc'])
                mfcc_scaled = scaler.transform(mfcc_flat)
                mfcc_input = mfcc_scaled.reshape(1, config['max_len'], config['n_mfcc'])
                
                pred = model.predict(mfcc_input)
                emotion = le.inverse_transform([np.argmax(pred)])[0]
                confidence = np.max(pred) * 100
                
                emoji_map = {'angry': '😠', 'disgust': '🤢', 'fear': '😨', 
                            'happy': '😊', 'neutral': '😐', 'pleasant_surprise': '😲', 'sad': '😢'}
                emoji = emoji_map.get(emotion, '🎭')
                
                st.success(f"**Detected Emotion: {emoji} {emotion.upper()}** (Confidence: {confidence:.1f}%)")
                st.bar_chart(pd.DataFrame(pred[0], index=le.classes_, columns=["Probability"]))
                
                os.unlink(tmp_path)
        
        if os.path.exists(f'{MODEL_DIR}/tess_comparison_results.csv'):
            st.subheader("Model Comparison")
            comp = pd.read_csv(f'{MODEL_DIR}/tess_comparison_results.csv', index_col=0)
            st.dataframe(comp, use_container_width=True)
    except Exception as e:
        st.warning(f"TESS models not loaded. Error: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Deep Learning Models: TensorFlow/Keras")

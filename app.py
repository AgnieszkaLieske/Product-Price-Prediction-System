import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json
import os
from pathlib import Path

# ====================================================================
# KONFIGURACJA ŚCIEŻEK - DLA STRUKTURY: app.py w katalogu głównym
# ====================================================================

# Katalog główny to tam gdzie jest app.py
BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path('.')
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

# ====================================================================
# KONFIGURACJA STRONY
# ====================================================================

st.set_page_config(
    page_title="Atlas - Wycena Maszyn",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .price-display {
        font-size: 3.5rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# FUNKCJE POMOCNICZE
# ====================================================================

@st.cache_resource
def load_model_components():
    """Wczytaj model, scaler i feature names"""
    try:
        # Debug info
        st.sidebar.caption(f"📂 BASE_DIR: {BASE_DIR}")
        st.sidebar.caption(f"📂 MODELS_DIR: {MODELS_DIR}")
        st.sidebar.caption(f"✅ Katalog istnieje: {MODELS_DIR.exists()}")
        
        model_path = MODELS_DIR / 'model_ridge.pkl'
        st.sidebar.caption(f"🔍 Szukam: {model_path}")
        st.sidebar.caption(f"✅ Plik istnieje: {model_path.exists()}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(MODELS_DIR / 'feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open(MODELS_DIR / 'features_to_scale.pkl', 'rb') as f:
            features_to_scale = pickle.load(f)
        
        try:
            with open(MODELS_DIR / 'training_metrics.json', 'r') as f:
                metrics = json.load(f)
        except:
            metrics = None
        
        st.sidebar.success("✅ Model załadowany!")
        return model, scaler, feature_names, features_to_scale, metrics
    
    except FileNotFoundError as e:
        st.error("❌ **BŁĄD: Nie znaleziono plików modelu!**")
        st.error(f"Szukano w: `{MODELS_DIR}`")
        st.error(f"Szczegóły: {e}")
        
        # Szczegółowa diagnostyka
        with st.expander("🔍 Sprawdź strukturę plików w repozytorium", expanded=True):
            st.code(f"Obecny katalog roboczy: {os.getcwd()}")
            st.code(f"__file__: {__file__ if '__file__' in globals() else 'BRAK'}")
            st.code(f"BASE_DIR: {BASE_DIR}")
            st.code(f"MODELS_DIR: {MODELS_DIR}")
            st.code(f"MODELS_DIR.exists(): {MODELS_DIR.exists()}")
            
            st.markdown("**📁 Pliki w katalogu głównym:**")
            try:
                for item in sorted(BASE_DIR.iterdir()):
                    icon = "📁" if item.is_dir() else "📄"
                    st.text(f"{icon} {item.name}")
            except Exception as ex:
                st.error(f"Nie można odczytać katalogu: {ex}")
            
            if MODELS_DIR.exists():
                st.markdown(f"**📁 Pliki w {MODELS_DIR.name}/:**")
                try:
                    for item in sorted(MODELS_DIR.iterdir()):
                        size = item.stat().st_size / 1024 / 1024  # MB
                        st.text(f"📄 {item.name} ({size:.2f} MB)")
                except Exception as ex:
                    st.error(f"Błąd odczytu: {ex}")
            else:
                st.error(f"❌ Katalog `{MODELS_DIR}` NIE ISTNIEJE!")
                st.markdown("**Możliwe przyczyny:**")
                st.markdown("- Katalog `models/` nie został dodany do repo GitHub")
                st.markdown("- Pliki `.pkl` są w `.gitignore`")
                st.markdown("- Nie wykonano `git push` po dodaniu plików")
        
        st.info("""
        **🔧 Co zrobić?**
        
        1️⃣ Sprawdź na GitHub czy folder `models/` jest widoczny
        2️⃣ Kliknij w folder - czy widzisz pliki .pkl?
        3️⃣ Jeśli NIE, wykonaj lokalnie:
        ```bash
        git add -f models/*.pkl
        git commit -m "Add model files"
        git push
        ```
        4️⃣ Poczekaj ~30 sek i odśwież aplikację
        """)
        st.stop()
    
    except Exception as e:
        st.error(f"❌ Inny błąd: {e}")
        st.exception(e)
        st.stop()

@st.cache_data
def load_specs():
    """Wczytaj specyfikacje modeli"""
    try:
        specs_path = DATA_DIR / 'material_handlers_specs.csv'
        specs = pd.read_csv(specs_path)
        return specs
    except FileNotFoundError:
        st.error(f"❌ Nie znaleziono: `{DATA_DIR / 'material_handlers_specs.csv'}`")
        
        with st.expander("🔍 Debug - pliki w data/"):
            if DATA_DIR.exists():
                for item in DATA_DIR.iterdir():
                    st.text(f"📄 {item.name}")
            else:
                st.error(f"Katalog {DATA_DIR} nie istnieje!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Błąd wczytywania CSV: {e}")
        st.stop()

def calculate_age(year, current_year=2025):
    """Oblicz wiek maszyny"""
    return current_year - year

def map_condition_pl_to_en(condition_pl):
    """Mapowanie polskich nazw stanów na angielskie"""
    mapping = {
        'Doskonały': 'excellent',
        'Bardzo dobry': 'good', 
        'Dobry': 'fair',
        'Zadowalający': 'poor',
        'Wymaga naprawy': 'very_poor'
    }
    return mapping.get(condition_pl, 'fair')

def prepare_input_features(
    selected_model, model_spec, age, hours, condition_pl,
    condition_score, full_service_history, major_repairs,
    has_ac, has_gps, extra_attachments, feature_names
):
    """Przygotuj features do predykcji"""
    
    input_dict = {feat: 0 for feat in feature_names}
    
    # Parametry numeryczne
    input_dict['tonnage'] = model_spec['tonnage']
    input_dict['engine_power_kw'] = model_spec['engine_power_kw']
    input_dict['reach_m'] = model_spec['reach_m']
    input_dict['base_price_new'] = model_spec['base_price_new']
    input_dict['age'] = age
    input_dict['hours'] = hours
    input_dict['hours_per_year'] = round(hours / age, 1) if age > 0 else 0
    input_dict['condition_score'] = condition_score
    input_dict['major_repairs'] = major_repairs
    input_dict['extra_attachments'] = extra_attachments
    
    # Features binarne
    input_dict['is_hybrid'] = int(model_spec['is_hybrid'])
    input_dict['full_service_history'] = int(full_service_history)
    input_dict['has_ac'] = int(has_ac)
    input_dict['has_gps'] = int(has_gps)
    
    # One-hot encoding
    model_column_name = f'model_{selected_model}'
    if model_column_name in input_dict:
        input_dict[model_column_name] = 1
    
    condition_en = map_condition_pl_to_en(condition_pl)
    condition_column_name = f'condition_{condition_en}'
    if condition_column_name in input_dict:
        input_dict[condition_column_name] = 1
    
    df = pd.DataFrame([input_dict])
    df = df[feature_names]
    
    return df

# ====================================================================
# WCZYTANIE DANYCH
# ====================================================================

model, scaler, feature_names, features_to_scale, metrics = load_model_components()
specs_df = load_specs()

# ====================================================================
# NAGŁÓWEK
# ====================================================================

st.markdown('<div class="main-header">🏗️ Atlas Poland - Wycena Maszyn</div>', unsafe_allow_html=True)
st.markdown("### 🤖 System AI do automatycznej wyceny maszyn używanych")

if metrics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Dokładność (R²)", f"{metrics['performance_metrics']['test']['r2']:.2%}")
    with col2:
        st.metric("📊 Błąd średni", f"{metrics['performance_metrics']['test']['mape']:.1f}%")
    with col3:
        st.metric("📈 Model", "Ridge Regression")
    with col4:
        st.metric("📅 Trenowanie", metrics['training_info']['date'][:10])

st.markdown("---")
st.info("💡 **Instrukcja:** Wypełnij formularz poniżej, wybierając parametry maszyny. System automatycznie obliczy szacowaną cenę.")

# ====================================================================
# FORMULARZ
# ====================================================================

with st.form("valuation_form"):
    st.subheader("📋 Formularz wyceny")
    
    # Model
    st.markdown("#### 🏗️ Wybór modelu")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_options = sorted(specs_df['model'].tolist())
        selected_model = st.selectbox("Model maszyny Atlas", options=model_options)
    
    model_spec = specs_df[specs_df['model'] == selected_model].iloc[0]
    
    with col2:
        st.markdown("**Parametry techniczne:**")
        st.caption(f"⚖️ Waga: {model_spec['tonnage']} ton")
        st.caption(f"⚡ Moc: {model_spec['engine_power_kw']} kW")
        st.caption(f"📏 Zasięg: {model_spec['reach_m']} m")
        st.caption(f"💰 Cena nowa: {model_spec['base_price_new']:,} zł")
        if model_spec['is_hybrid']:
            st.success("🔋 Wersja hybrydowa")
    
    st.markdown("---")
    
    # Dane użytkowania
    st.markdown("#### 📅 Dane użytkowania")
    col3, col4 = st.columns(2)
    
    with col3:
        year = st.slider("Rok produkcji", min_value=2010, max_value=2024, value=2018)
        age = calculate_age(year)
        st.info(f"🕐 Wiek maszyny: **{age} {'rok' if age == 1 else 'lata' if age < 5 else 'lat'}**")
    
    with col4:
        hours = st.number_input("Motogodziny", min_value=0, max_value=25000, value=5000, step=100)
        hours_per_year = round(hours / age, 1) if age > 0 else 0
        st.caption(f"📊 Średnie: **{hours_per_year} godz/rok**")
    
    st.markdown("---")
    
    # Stan techniczny
    st.markdown("#### 🔧 Stan techniczny")
    col5, col6 = st.columns(2)
    
    with col5:
        condition = st.radio(
            "Stan ogólny",
            options=['Doskonały', 'Bardzo dobry', 'Dobry', 'Zadowalający', 'Wymaga naprawy'],
            index=2
        )
        
        condition_score_map = {
            'Doskonały': 4, 'Bardzo dobry': 3, 'Dobry': 2,
            'Zadowalający': 1, 'Wymaga naprawy': 0
        }
        condition_score = condition_score_map[condition]
    
    with col6:
        full_service_history = st.checkbox("✅ Pełna historia serwisowa", value=True)
        major_repairs = st.slider("Liczba napraw głównych", 0, 5, 0)
    
    st.markdown("---")
    
    # Wyposażenie
    st.markdown("#### 🛠️ Wyposażenie")
    col7, col8 = st.columns(2)
    
    with col7:
        has_ac = st.checkbox("❄️ Klimatyzacja", value=True)
        has_gps = st.checkbox("📡 GPS / Telematyka", value=False)
    
    with col8:
        extra_attachments = st.slider("Dodatkowe osprzęty", 0, 3, 1)
    
    st.markdown("---")
    submitted = st.form_submit_button("💰 WYCEN MASZYNĘ", use_container_width=True, type="primary")

# ====================================================================
# PREDYKCJA
# ====================================================================

if submitted:
    st.markdown("---")
    st.markdown("## 📊 Wynik wyceny")
    
    with st.spinner("🔄 Obliczam cenę..."):
        try:
            X_input = prepare_input_features(
                selected_model, model_spec, age, hours, condition,
                condition_score, full_service_history, major_repairs,
                has_ac, has_gps, extra_attachments, feature_names
            )
            
            X_input_scaled = X_input.copy()
            X_input_scaled[features_to_scale] = scaler.transform(X_input[features_to_scale])
            
            predicted_price = model.predict(X_input_scaled)[0]
            
            if predicted_price < 0:
                predicted_price = 10000
            
            confidence_interval_pct = metrics['performance_metrics']['test']['mape'] / 100 if metrics else 0.12
            lower_bound = predicted_price * (1 - confidence_interval_pct)
            upper_bound = predicted_price * (1 + confidence_interval_pct)
            
            # Wyświetlenie
            st.markdown(f'<div class="price-display">{predicted_price:,.0f} PLN</div>', unsafe_allow_html=True)
            
            st.markdown("### 📈 Szczegóły wyceny")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric("💰 Cena szacowana", f"{predicted_price:,.0f} zł")
            
            with col_m2:
                st.metric(
                    "📊 Przedział cenowy",
                    f"±{confidence_interval_pct*100:.0f}%",
                    f"{lower_bound:,.0f} - {upper_bound:,.0f} zł"
                )
            
            with col_m3:
                confidence = metrics['performance_metrics']['test']['r2'] if metrics else 0.87
                st.metric("🎯 Pewność", f"{confidence:.0%}")
            
            # Rekomendacje
            st.markdown("---")
            st.markdown("### 💡 Rekomendacje handlowe")
            
            col_rec1, col_rec2 = st.columns(2)
            
            buy_min = lower_bound * 0.90
            buy_max = predicted_price * 0.95
            sell_min = predicted_price * 1.05
            sell_max = upper_bound * 1.10
            
            with col_rec1:
                st.success(f"""
                **📥 ZAKUP:**
                
                💵 Min: **{buy_min:,.0f} zł**  
                💰 Max: **{buy_max:,.0f} zł**
                """)
            
            with col_rec2:
                st.info(f"""
                **📤 SPRZEDAŻ:**
                
                💵 Min: **{sell_min:,.0f} zł**  
                💰 Target: **{sell_max:,.0f} zł**
                """)
            
            margin = sell_min - buy_max
            margin_pct = (margin / buy_max) * 100
            st.success(f"💹 **Potencjalna marża:** {margin:,.0f} zł ({margin_pct:.1f}%)")
            
        except Exception as e:
            st.error("❌ Wystąpił błąd podczas wyceny")
            st.exception(e)

# ====================================================================
# SIDEBAR
# ====================================================================

with st.sidebar:
    st.markdown("### ℹ️ O systemie")
    st.markdown("System AI do wyceny maszyn przeładunkowych Atlas.")
    
    if metrics:
        st.markdown("### 📊 Parametry modelu")
        st.metric("Dokładność", f"{metrics['performance_metrics']['test']['r2']:.1%}")
        st.metric("Błąd MAPE", f"{metrics['performance_metrics']['test']['mape']:.1f}%")
    
    st.markdown("---")
    st.caption("© 2025 Atlas Poland")

st.markdown("---")
st.caption("🤖 Wycena AI | Dokładność: ~89% R²")

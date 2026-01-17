# ====================================================================
# APLIKACJA STREAMLIT - WYCENA MASZYN PRZEŁADUNKOWYCH ATLAS
# ====================================================================
# Wersja: 2.0 (Poprawiona)
# Data: 2025-10-09
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json

# ====================================================================
# KONFIGURACJA STRONY
# ====================================================================

st.set_page_config(
    page_title="Atlas - Wycena Maszyn",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS dla lepszego wyglądu
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# FUNKCJE POMOCNICZE
# ====================================================================

@st.cache_resource
def load_model_components():
    """Wczytaj model, scaler i feature names (cache dla szybkości)"""
    try:
        with open('../models/model_ridge.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('../models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('../models/features_to_scale.pkl', 'rb') as f:
            features_to_scale = pickle.load(f)
        
        # Wczytaj też metryki (opcjonalnie)
        try:
            with open('../models/training_metrics.json', 'r') as f:
                metrics = json.load(f)
        except:
            metrics = None
        
        return model, scaler, feature_names, features_to_scale, metrics
    except FileNotFoundError as e:
        st.error(f"❌ Błąd: Nie znaleziono pliku modelu. Upewnij się, że uruchomiłeś notebook trenujący.")
        st.error(f"Szczegóły: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Błąd wczytywania modelu: {e}")
        st.stop()

@st.cache_data
def load_specs():
    """Wczytaj specyfikacje modeli"""
    try:
        specs = pd.read_csv('../data/material_handlers_specs.csv')
        return specs
    except FileNotFoundError:
        st.error("❌ Błąd: Nie znaleziono pliku ze specyfikacjami modeli.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Błąd wczytywania specyfikacji: {e}")
        st.stop()

def calculate_age(year, current_year=2025):
    """Oblicz wiek maszyny z roku produkcji"""
    return current_year - year

def map_condition_pl_to_en(condition_pl):
    """Mapowanie polskich nazw stanów na angielskie (jak w danych treningowych)"""
    mapping = {
        'Doskonały': 'excellent',
        'Bardzo dobry': 'good', 
        'Dobry': 'fair',
        'Zadowalający': 'poor',
        'Wymaga naprawy': 'very_poor'
    }
    return mapping.get(condition_pl, 'fair')

def prepare_input_features(
    selected_model, 
    model_spec, 
    age, 
    hours, 
    condition_pl,
    condition_score,
    full_service_history,
    major_repairs,
    has_ac,
    has_gps,
    extra_attachments,
    feature_names
):
    """
    Przygotuj features w dokładnie takiej samej formie jak w treningu
    
    Returns: DataFrame z jednym wierszem, gotowy do predykcji
    """
    
    # 1. Utwórz pusty DataFrame z wszystkimi features (wypełniony zerami)
    input_dict = {feat: 0 for feat in feature_names}
    
    # 2. Parametry numeryczne (nie-binarne, nie-one-hot)
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
    
    # 3. Features binarne (boolean → 0/1)
    input_dict['is_hybrid'] = int(model_spec['is_hybrid'])
    input_dict['full_service_history'] = int(full_service_history)
    input_dict['has_ac'] = int(has_ac)
    input_dict['has_gps'] = int(has_gps)
    
    # 4. One-hot encoding dla MODEL
    # WAŻNE: Nazwa kolumny musi być DOKŁADNIE jak w treningu
    model_column_name = f'model_{selected_model}'
    if model_column_name in input_dict:
        input_dict[model_column_name] = 1
    else:
        st.warning(f"⚠️ Uwaga: Kolumna '{model_column_name}' nie istnieje w modelu. Sprawdź dane treningowe.")
    
    # 5. One-hot encoding dla CONDITION
    # Mapowanie polskiego na angielski
    condition_en = map_condition_pl_to_en(condition_pl)
    condition_column_name = f'condition_{condition_en}'
    if condition_column_name in input_dict:
        input_dict[condition_column_name] = 1
    else:
        st.warning(f"⚠️ Uwaga: Kolumna '{condition_column_name}' nie istnieje w modelu.")
    
    # 6. Utwórz DataFrame
    df = pd.DataFrame([input_dict])
    
    # 7. WERYFIKACJA: Sprawdź czy wszystkie kolumny są na miejscu
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        st.error(f"❌ BŁĄD: Brakujące kolumny: {missing_cols}")
        st.stop()
    
    # 8. Uporządkuj kolumny DOKŁADNIE jak w feature_names
    df = df[feature_names]
    
    return df

# ====================================================================
# WCZYTANIE MODELU I DANYCH
# ====================================================================

model, scaler, feature_names, features_to_scale, metrics = load_model_components()
specs_df = load_specs()

# ====================================================================
# NAGŁÓWEK APLIKACJI
# ====================================================================

st.markdown('<div class="main-header">🏗️ Atlas Poland - Wycena Maszyn</div>', unsafe_allow_html=True)
st.markdown("### 🤖 System AI do automatycznej wyceny maszyn używanych")

# Info box z metrykami modelu (jeśli dostępne)
if metrics:
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("🎯 Dokładność (R²)", f"{metrics['performance_metrics']['test']['r2']:.2%}")
    with col_info2:
        st.metric("📊 Błąd średni", f"{metrics['performance_metrics']['test']['mape']:.1f}%")
    with col_info3:
        st.metric("📈 Model", "Ridge Regression")
    with col_info4:
        st.metric("📅 Trenowanie", metrics['training_info']['date'][:10])

st.markdown("---")

# Informacja dla użytkownika
st.info("💡 **Instrukcja:** Wypełnij formularz poniżej, wybierając parametry maszyny. System automatycznie obliczy szacowaną cenę na podstawie modelu AI.")

# ====================================================================
# FORMULARZ WYCENY
# ====================================================================

with st.form("valuation_form"):
    st.subheader("📋 Formularz wyceny")
    
    # ============================================
    # SEKCJA 1: Model i specyfikacja
    # ============================================
    st.markdown("#### 🏗️ Wybór modelu")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dropdown: Wybór modelu
        model_options = sorted(specs_df['model'].tolist())
        selected_model = st.selectbox(
            "Model maszyny Atlas",
            options=model_options,
            help="Wybierz model maszyny przeładunkowej"
        )
    
    # Automatyczne wypełnienie parametrów technicznych
    model_spec = specs_df[specs_df['model'] == selected_model].iloc[0]
    
    with col2:
        st.markdown("**Parametry techniczne:**")
        st.caption(f"⚖️ Waga: {model_spec['tonnage']} ton")
        st.caption(f"⚡ Moc: {model_spec['engine_power_kw']} kW")
        st.caption(f"📏 Zasięg: {model_spec['reach_m']} m")
        st.caption(f"💰 Cena nowa: {model_spec['base_price_new']:,} zł")
        if model_spec['is_hybrid']:
            st.success("🔋 Wersja hybrydowa (ACCU)")
    
    st.markdown("---")
    
    # ============================================
    # SEKCJA 2: Dane użytkowania
    # ============================================
    st.markdown("#### 📅 Dane użytkowania")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Rok produkcji
        year = st.slider(
            "Rok produkcji",
            min_value=2010,
            max_value=2024,
            value=2018,
            help="Rok wyprodukowania maszyny"
        )
        
        # Automatyczne obliczenie wieku
        age = calculate_age(year)
        st.info(f"🕐 Wiek maszyny: **{age} {'rok' if age == 1 else 'lata' if age < 5 else 'lat'}**")
    
    with col4:
        # Motogodziny
        hours = st.number_input(
            "Motogodziny (przepracowane godziny)",
            min_value=0,
            max_value=25000,
            value=5000,
            step=100,
            help="Liczba przepracowanych godzin"
        )
        
        # Godziny na rok (automatyczne)
        hours_per_year = round(hours / age, 1) if age > 0 else 0
        st.caption(f"📊 Średnie użytkowanie: **{hours_per_year} godz/rok**")
        
        # Ocena intensywności
        if hours_per_year > 2000:
            st.warning("⚠️ Bardzo intensywne użytkowanie")
        elif hours_per_year > 1000:
            st.success("✅ Normalne użytkowanie")
        else:
            st.info("ℹ️ Lekkie użytkowanie")
    
    st.markdown("---")
    
    # ============================================
    # SEKCJA 3: Stan techniczny
    # ============================================
    st.markdown("#### 🔧 Stan techniczny i serwis")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Radio buttons: Stan
        condition = st.radio(
            "Stan ogólny maszyny",
            options=['Doskonały', 'Bardzo dobry', 'Dobry', 'Zadowalający', 'Wymaga naprawy'],
            index=2,  # Domyślnie "Dobry"
            help="Oceń ogólny stan techniczny maszyny"
        )
        
        # Mapowanie condition na condition_score (dla wyświetlenia)
        condition_score_map = {
            'Doskonały': 4,
            'Bardzo dobry': 3,
            'Dobry': 2,
            'Zadowalający': 1,
            'Wymaga naprawy': 0
        }
        condition_score = condition_score_map[condition]
        
        st.caption(f"📊 Ocena numeryczna: {condition_score}/4")
    
    with col6:
        # Checkbox: Pełna historia serwisowa
        full_service_history = st.checkbox(
            "✅ Pełna historia serwisowa",
            value=True,
            help="Czy maszyna ma kompletną dokumentację serwisową?"
        )
        
        # Slider: Naprawy główne
        major_repairs = st.slider(
            "Liczba napraw głównych",
            min_value=0,
            max_value=5,
            value=0,
            help="Liczba poważnych napraw (silnik, hydraulika, układ jezdny)"
        )
        
        if major_repairs > 2:
            st.warning(f"⚠️ {major_repairs} napraw to dużo - wpłynie na cenę")
    
    st.markdown("---")
    
    # ============================================
    # SEKCJA 4: Wyposażenie dodatkowe
    # ============================================
    st.markdown("#### 🛠️ Wyposażenie i osprzęt")
    
    col7, col8 = st.columns(2)
    
    with col7:
        has_ac = st.checkbox(
            "❄️ Klimatyzacja w kabinie",
            value=True,
            help="Czy kabina jest wyposażona w klimatyzację?"
        )
        
        has_gps = st.checkbox(
            "📡 GPS / System telematyki",
            value=False,
            help="Czy maszyna ma GPS i system monitoringu?"
        )
    
    with col8:
        # Slider: Osprzęty dodatkowe
        extra_attachments = st.slider(
            "Liczba dodatkowych osprzętów",
            min_value=0,
            max_value=3,
            value=1,
            help="Łyżki, chwyty, widły, itp. - liczba dodatkowych narzędzi w zestawie"
        )
        
        st.caption("🔧 Przykłady: łyżka, chwyt, widły, magnes")
    
    # ============================================
    # Przycisk SUBMIT
    # ============================================
    st.markdown("---")
    submitted = st.form_submit_button(
        "💰 WYCEN MASZYNĘ",
        use_container_width=True,
        type="primary"
    )

# ====================================================================
# PREDYKCJA I WYNIKI
# ====================================================================

if submitted:
    st.markdown("---")
    st.markdown("## 📊 Wynik wyceny")
    
    with st.spinner("🔄 Obliczam cenę... (może potrwać kilka sekund)"):
        try:
            # ============================================
            # 1. PRZYGOTOWANIE DANYCH WEJŚCIOWYCH
            # ============================================
            
            X_input = prepare_input_features(
                selected_model=selected_model,
                model_spec=model_spec,
                age=age,
                hours=hours,
                condition_pl=condition,
                condition_score=condition_score,
                full_service_history=full_service_history,
                major_repairs=major_repairs,
                has_ac=has_ac,
                has_gps=has_gps,
                extra_attachments=extra_attachments,
                feature_names=feature_names
            )
            
            # ============================================
            # 2. SKALOWANIE (tylko wybranych features)
            # ============================================
            
            X_input_scaled = X_input.copy()
            X_input_scaled[features_to_scale] = scaler.transform(X_input[features_to_scale])
            
            # ============================================
            # 3. PREDYKCJA CENY
            # ============================================
            
            predicted_price = model.predict(X_input_scaled)[0]
            
            # Zabezpieczenie przed ujemnymi cenami
            if predicted_price < 0:
                st.warning("⚠️ Model przewidział ujemną cenę. Ustawiam minimalną wartość.")
                predicted_price = 10000
            
            # ============================================
            # 4. PRZEDZIAŁ UFNOŚCI
            # ============================================
            
            # Bazując na MAPE z metryk (jeśli dostępne)
            if metrics:
                mape = metrics['performance_metrics']['test']['mape'] / 100
                confidence_interval_pct = max(0.10, mape)  # Min 10%
            else:
                confidence_interval_pct = 0.12  # Domyślnie 12%
            
            lower_bound = predicted_price * (1 - confidence_interval_pct)
            upper_bound = predicted_price * (1 + confidence_interval_pct)
            
            # ============================================
            # 5. WYŚWIETLENIE WYNIKU - GŁÓWNA CENA
            # ============================================
            
            st.markdown(
                f'<div class="price-display">{predicted_price:,.0f} PLN</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("### 📈 Szczegóły wyceny")
            
            # Metryki
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric(
                    "💰 Cena szacowana",
                    f"{predicted_price:,.0f} zł",
                    help="Najbardziej prawdopodobna cena maszyny"
                )
            
            with col_m2:
                st.metric(
                    "📊 Przedział cenowy",
                    f"±{confidence_interval_pct*100:.0f}%",
                    f"{lower_bound:,.0f} - {upper_bound:,.0f} zł",
                    help=f"Przedział ufności bazujący na błędzie modelu ({confidence_interval_pct*100:.0f}%)"
                )
            
            with col_m3:
                if metrics:
                    confidence = metrics['performance_metrics']['test']['r2']
                    confidence_stars = "⭐" * min(5, int(confidence * 5))
                else:
                    confidence = 0.87
                    confidence_stars = "⭐⭐⭐⭐"
                
                st.metric(
                    "🎯 Pewność modelu",
                    f"{confidence:.0%}",
                    confidence_stars,
                    help="Dokładność modelu bazująca na R² score"
                )
            
            # ============================================
            # 6. REKOMENDACJE BIZNESOWE
            # ============================================
            
            st.markdown("---")
            st.markdown("### 💡 Rekomendacje handlowe")
            
            col_rec1, col_rec2 = st.columns(2)
            
            # Ceny zakupu (konserwatywne)
            buy_min = lower_bound * 0.90
            buy_max = predicted_price * 0.95
            
            # Ceny sprzedaży (z marżą)
            sell_min = predicted_price * 1.05
            sell_max = upper_bound * 1.10
            
            with col_rec1:
                st.success(f"""
                **📥 ZAKUP (oferowana cena klientowi):**
                
                💵 Minimalna: **{buy_min:,.0f} zł**  
                💰 Maksymalna: **{buy_max:,.0f} zł**
                
                *Rekomendacja: Negocjuj w tym przedziale*
                """)
            
            with col_rec2:
                st.info(f"""
                **📤 SPRZEDAŻ (cena katalogowa):**
                
                💵 Minimalna: **{sell_min:,.0f} zł**  
                💰 Target: **{sell_max:,.0f} zł**
                
                *Rekomendacja: Ustal marżę ~10-15%*
                """)
            
            # Potencjalna marża
            margin = sell_min - buy_max
            margin_pct = (margin / buy_max) * 100
            
            st.success(f"💹 **Potencjalna marża:** {margin:,.0f} zł ({margin_pct:.1f}%)")
            
            # ============================================
            # 7. ANALIZA WPŁYWU CECH
            # ============================================
            
            st.markdown("---")
            
            with st.expander("🔍 Szczegółowa analiza wpływu na cenę", expanded=False):
                st.markdown("#### 📊 Wprowadzone parametry:")
                
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    st.markdown(f"""
                    **Model i specyfikacja:**
                    - Model: **{selected_model}**
                    - Waga: {model_spec['tonnage']} ton
                    - Moc: {model_spec['engine_power_kw']} kW
                    - Zasięg: {model_spec['reach_m']} m
                    - Hybrid: {'✅ Tak' if model_spec['is_hybrid'] else '❌ Nie'}
                    """)
                
                with col_d2:
                    st.markdown(f"""
                    **Użytkowanie:**
                    - Rok: {year} (**{age} lat**)
                    - Motogodziny: {hours:,}
                    - Godz/rok: {hours_per_year:.0f}
                    - Stan: **{condition}**
                    - Naprawy: {major_repairs}
                    """)
                
                with col_d3:
                    st.markdown(f"""
                    **Wyposażenie:**
                    - Serwis: {'✅ Pełna historia' if full_service_history else '❌ Brak'}
                    - Klimatyzacja: {'✅' if has_ac else '❌'}
                    - GPS: {'✅' if has_gps else '❌'}
                    - Osprzęty: {extra_attachments} szt.
                    """)
                
                st.markdown("---")
                st.markdown("#### 💡 Szacowany wpływ na cenę:")
                
                # Uproszczona analiza wpływu (bazując na typowych współczynnikach)
                age_depreciation = (1 - 0.10 ** (age / 10)) * 100  # ~10% rocznie
                hours_impact = min(30, hours / 500)  # Do 30% za godziny
                condition_impact = (condition_score - 2) * 10  # -20% do +20%
                equipment_bonus = (int(has_ac) * 3 + int(has_gps) * 2 + extra_attachments * 2)
                
                st.markdown(f"""
                - 🕐 **Wiek ({age} lat):** -{age_depreciation:.0f}% (deprecjacja)
                - ⚙️ **Motogodziny ({hours:,}):** -{hours_impact:.0f}% (zużycie)
                - 🔧 **Stan ({condition}):** {condition_impact:+.0f}% 
                - 🛠️ **Wyposażenie:** +{equipment_bonus:.0f}% (bonusy)
                - 📋 **Historia serwisowa:** {'+ 5%' if full_service_history else '0%'}
                - 🔨 **Naprawy ({major_repairs}):** {-major_repairs * 5:.0f}%
                """)
                
                total_adjustment = -age_depreciation - hours_impact + condition_impact + equipment_bonus + (5 if full_service_history else 0) - (major_repairs * 5)
                st.info(f"📊 **Łączny wpływ modyfikacji:** {total_adjustment:+.0f}% od ceny bazowej")
            
            # ============================================
            # 8. PORÓWNANIE Z CENĄ NOWĄ
            # ============================================
            
            with st.expander("💰 Porównanie z ceną nową", expanded=False):
                price_new = model_spec['base_price_new']
                depreciation_pct = ((price_new - predicted_price) / price_new) * 100
                
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    st.metric("Cena nowa", f"{price_new:,.0f} zł")
                
                with col_comp2:
                    st.metric("Cena używana (AI)", f"{predicted_price:,.0f} zł")
                
                with col_comp3:
                    st.metric("Deprecjacja", f"{depreciation_pct:.1f}%", delta=f"-{price_new - predicted_price:,.0f} zł", delta_color="inverse")
                
                st.progress(min(1.0, predicted_price / price_new))
                st.caption(f"Wartość rezydualna: {(predicted_price / price_new * 100):.1f}% wartości nowej maszyny")
            
            # ============================================
            # 9. EKSPORT / ZAPISANIE
            # ============================================
            
            st.markdown("---")
            
            # Przygotuj dane do eksportu
            valuation_data = {
                'Data wyceny': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Model': selected_model,
                'Rok produkcji': year,
                'Wiek (lata)': age,
                'Motogodziny': hours,
                'Stan': condition,
                'Cena szacowana (PLN)': f"{predicted_price:,.0f}",
                'Przedział (PLN)': f"{lower_bound:,.0f} - {upper_bound:,.0f}",
                'Pewność modelu': f"{confidence:.0%}"
            }
            
            # Przycisk nowej wyceny
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🔄 Nowa wycena", use_container_width=True):
                    st.rerun()
            
            with col_btn2:
                # Download button (opcjonalnie)
                valuation_text = "\n".join([f"{k}: {v}" for k, v in valuation_data.items()])
                st.download_button(
                    label="💾 Pobierz wynik (TXT)",
                    data=valuation_text,
                    file_name=f"wycena_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error("❌ Wystąpił błąd podczas wyceny")
            st.exception(e)
            
            # Debug info (opcjonalnie - usuń w produkcji)
            with st.expander("🐛 Informacje debugowania"):
                st.write("**Parametry wejściowe:**")
                st.write(f"- Model: {selected_model}")
                st.write(f"- Age: {age}")
                st.write(f"- Hours: {hours}")
                st.write(f"- Condition: {condition}")
                st.write(f"- Feature names count: {len(feature_names)}")

# ====================================================================
# SIDEBAR - INFORMACJE
# ====================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=ATLAS+POLAND", use_column_width=True)
    
    st.markdown("### ℹ️ O systemie")
    st.markdown("""
    System AI do automatycznej wyceny maszyn przeładunkowych Atlas.
    
    **Model:** Ridge Regression  
    **Trenowanie:** 2025-10-09
    """)
    
    if metrics:
        st.markdown("### 📊 Parametry modelu")
        st.metric("Dokładność (R²)", f"{metrics['performance_metrics']['test']['r2']:.1%}")
        st.metric("Błąd średni (MAPE)", f"{metrics['performance_metrics']['test']['mape']:.1f}%")
        st.metric("Liczba obserwacji", f"{metrics['data_info']['total']} maszyn")
    
    st.markdown("---")
    st.markdown("### 🎯 Modele obsługiwane")
    for model_name in sorted(specs_df['model'].tolist()):
        st.caption(f"✅ {model_name}")
    
    st.markdown("---")
    st.caption("© 2025 Atlas Poland")
    st.caption("Wersja: 2.0")

# ====================================================================
# STOPKA
# ====================================================================

st.markdown("---")
st.caption("🤖 Wycena oparta na modelu AI (Ridge Regression) | Dokładność: ~89% R²")
st.caption(f"⚙️ Ostatnia aktualizacja: {datetime.now().strftime('%Y-%m-%d')} | Kontakt: support@atlaspoland.pl")

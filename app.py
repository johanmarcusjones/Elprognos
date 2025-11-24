import streamlit as st
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import joblib
import holidays
import plotly.graph_objects as go # <--- NYTT VERKTYG
from datetime import datetime, timedelta

# --- INSTÄLLNINGAR ---
st.set_page_config(page_title="Elprognos SE4", page_icon="⚡", layout="wide")

LAT_SE = 55.605
LON_SE = 13.003
LAT_DE = 53.551
LON_DE = 9.993
VALUTAKURS = 11.60 
ANTAL_DAGAR = 5  # Hur många dagar framåt vi visar

# Gränser för färgläggning (Öre)
GRANS_BILLIGT = 40
GRANS_DYRT = 120

# --- DATAHÄMTNING ---
@st.cache_data
def hamta_vader_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": [LAT_SE, LAT_DE],
        "longitude": [LON_SE, LON_DE],
        "hourly": ["temperature_2m", "wind_speed_10m"],
        "timezone": "Europe/Berlin",
        "forecast_days": ANTAL_DAGAR + 2,
        "past_days": 3 # Hämta historia för att binda ihop grafen
    }
    
    responses = openmeteo.weather_api(url, params=params)

    # Malmö
    r1 = responses[0].Hourly()
    d1 = {
        "Tid": pd.date_range(
            start = pd.to_datetime(r1.Time(), unit = "s", utc = True),
            end = pd.to_datetime(r1.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = r1.Interval()),
            inclusive = "left"
        ).tz_convert('Europe/Stockholm'),
        "Temp_SE": r1.Variables(0).ValuesAsNumpy(),
        "Vind_SE": r1.Variables(1).ValuesAsNumpy()
    }
    df1 = pd.DataFrame(d1)

    # Hamburg
    r2 = responses[1].Hourly()
    d2 = {
        "Tid": df1['Tid'],
        "Vind_DE": r2.Variables(1).ValuesAsNumpy()
    }
    df2 = pd.DataFrame(d2)

    return pd.merge(df1, df2, on='Tid')

def ladda_modell():
    try:
        return joblib.load('min_el_modell.pkl')
    except:
        return None

# --- HUVUDPROGRAM ---
st.title("⚡ Elprognos SE4")
st.markdown("Prognos baserad på väderdata i **Malmö** och **Hamburg**.")

model = ladda_modell()

if model is None:
    st.error("⚠️ Saknar modellfilen! Ladda upp 'min_el_modell.pkl' till GitHub.")
else:
    with st.spinner('Räknar ut prognosen...'):
        df = hamta_vader_data()

    # Förbered AI-data
    df['Timme'] = df['Tid'].dt.hour
    df['Veckodag'] = df['Tid'].dt.dayofweek
    svenska_helgdagar = holidays.SE()
    df['RodDag'] = df['Tid'].apply(lambda x: 1 if x in svenska_helgdagar else 0)

    # Gissa priset
    X = df[['Temp_SE', 'Vind_SE', 'Vind_DE', 'Timme', 'Veckodag', 'RodDag']]
    pris_eur = model.predict(X)
    df['Pris_Sek'] = ((pris_eur * VALUTAKURS) / 10) * 1.25

    # Filtrera data för visning (Från igår till +5 dagar)
    nu = pd.Timestamp.now(tz='Europe/Stockholm')
    start_visning = nu - pd.Timedelta(hours=24)
    slut_visning = nu + pd.Timedelta(days=ANTAL_DAGAR)
    
    df_plot = df[(df['Tid'] >= start_visning) & (df['Tid'] <= slut_visning)].copy()

    # --- SKAPA DEN SNYGGA GRAFEN (PLOTLY) ---
    fig = go.Figure()

    # 1. Lägg till Priskurvan (Snygg linje med gradient-fyllning under)
    fig.add_trace(go.Scatter(
        x=df_plot['Tid'], 
        y=df_plot['Pris_Sek'],
        mode='lines',
        name='Elpris',
        line=dict(color='#00d4ff', width=3), # Neon-blå färg
        fill='tozeroy', # Fyll färg under linjen
        fillcolor='rgba(0, 212, 255, 0.1)' # Svag blå toning
    ))

    # 2. Lägg till vertikal linje för "NU"
    fig.add_vline(x=nu.timestamp() * 1000, line_width=2, line_dash="dash", line_color="white")
    fig.add_annotation(x=nu, y=df_plot['Pris_Sek'].max(), text="NU", showarrow=False, yshift=10, font=dict(color="white"))

    # 3. Färglägg bakgrunden (Zoner)
    # Billigt (Grönt)
    fig.add_hrect(y0=0, y1=GRANS_BILLIGT, line_width=0, fillcolor="green", opacity=0.1, layer="below")
    # Dyrt (Rött)
    fig.add_hrect(y0=GRANS_DYRT, y1=500, line_width=0, fillcolor="red", opacity=0.1, layer="below")

    # 4. Layout-inställningar (Dark Mode & Snygghet)
    fig.update_layout(
        template='plotly_dark', # Mörkt tema
        xaxis_title=None,
        yaxis_title="Öre/kWh",
        hovermode="x unified", # Snygg "crosshair" när man hovrar
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left", yanchor="bottom"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )

    # Visa grafen
    st.plotly_chart(fig, use_container_width=True)

    # --- BÄSTA LADDTIDERNA (KORT) ---
    st.markdown("### 🔌 Bästa laddtiderna (Kommande 3 dygn)")
    
    # Hitta bara framtida tider
    df_framtid = df[df['Tid'] > nu]
    
    cols = st.columns(3) # Tre kolumner för snygg layout
    
    # Sortera ut 3 billigaste
    basta = df_framtid.nsmallest(3, 'Pris_Sek')
    
    for i, (index, row) in enumerate(basta.iterrows()):
        dag = row['Tid'].strftime('%A') # t.ex "Monday"
        # Översätt dag
        dagar_map = {'Monday':'Mån', 'Tuesday':'Tis', 'Wednesday':'Ons', 'Thursday':'Tors', 'Friday':'Fre', 'Saturday':'Lör', 'Sunday':'Sön'}
        dag_sv = dagar_map.get(dag, dag)
        klocka = row['Tid'].strftime('%H:%M')
        pris = f"{row['Pris_Sek']:.1f}"
        
        # Visa snygga kort
        cols[i].metric(label=f"{dag_sv} {klocka}", value=f"{pris} öre")
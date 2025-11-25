import streamlit as st
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import joblib
import holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- INSTÄLLNINGAR ---
st.set_page_config(page_title="Elpris SE4", page_icon="⚡")

LAT_SE = 55.605
LON_SE = 13.003
LAT_DE = 53.551
LON_DE = 9.993
VALUTAKURS = 11.60 
ANTAL_DAGAR = 3

# Färggränser för staplarna
GRANS_LIG = 40   # Under detta = Grönt
GRANS_HOG = 100  # Över detta = Rött (Mellan = Gult)

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
        "past_days": 2 
    }
    
    responses = openmeteo.weather_api(url, params=params)

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

    r2 = responses[1].Hourly()
    d2 = {
        "Tid": df1['Tid'],
        "Vind_DE": r2.Variables(1).ValuesAsNumpy()
    }
    return pd.merge(df1, pd.DataFrame(d2), on='Tid')

def ladda_modell():
    try:
        return joblib.load('min_el_modell.pkl')
    except:
        return None

# --- FUNKTION FÖR ATT RITA STAPELGRAFEN ---
def rita_staplar(df_dag):
    # Skapa färger baserat på pris
    colors = []
    for pris in df_dag['Pris_Sek']:
        if pris < GRANS_LIG:
            colors.append('#00cc96') # Grön
        elif pris > GRANS_HOG:
            colors.append('#ef553b') # Röd
        else:
            colors.append('#fecb52') # Gul/Orange

    # Skapa stapeldiagram
    fig = go.Figure(go.Bar(
        x=df_dag['Pris_Sek'],
        y=df_dag['Tid_Timme'], # Visar "00-01", "01-02"
        orientation='h', # Horisontell
        marker_color=colors,
        text=df_dag['Pris_Sek'].round(0).astype(int).astype(str) + " öre", # Text i stapeln
        textposition='auto',
        hoverinfo='y+x'
    ))

    fig.update_layout(
        template='plotly_dark',
        height=len(df_dag) * 35, # Dynamisk höjd
        margin=dict(l=0, r=0, t=0, b=0),
        barmode='stack',
        yaxis=dict(autorange="reversed"), # 00:00 högst upp
        xaxis=dict(visible=False), # Dölj x-axeln (siffrorna står ju i stapeln)
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- APP START ---
st.markdown("### ⚡ Elpris SE4")

model = ladda_modell()

if model is None:
    st.error("⚠️ Ladda upp modellfilen!")
else:
    df = hamta_vader_data()

    # AI Process
    df['Timme'] = df['Tid'].dt.hour
    df['Veckodag'] = df['Tid'].dt.dayofweek
    svenska_helgdagar = holidays.SE()
    df['RodDag'] = df['Tid'].apply(lambda x: 1 if x in svenska_helgdagar else 0)

    X = df[['Temp_SE', 'Vind_SE', 'Vind_DE', 'Timme', 'Veckodag', 'RodDag']]
    pris_eur = model.predict(X)
    df['Pris_Sek'] = ((pris_eur * VALUTAKURS) / 10) * 1.25

    # Filter för Översikten (48h)
    nu = pd.Timestamp.now(tz='Europe/Stockholm')
    nu_timme = nu.floor('h')
    slut = nu + pd.Timedelta(hours=48)
    df_plot = df[(df['Tid'] >= nu_timme) & (df['Tid'] <= slut)].copy()

    aktuellt_pris = df_plot.iloc[0]['Pris_Sek'] if len(df_plot) > 0 else 0
    
    # 1. ÖVERST: Minimalistisk KPI & Linjegraf
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Just nu", value=f"{aktuellt_pris:.0f} öre")
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_plot['Tid'], y=df_plot['Pris_Sek'],
        mode='lines', line=dict(color='#29b6f6', width=2, shape='hv'),
        fill='tozeroy', fillcolor='rgba(41, 182, 246, 0.1)'
    ))
    fig_line.update_layout(
        template='plotly_dark', height=250, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, 
        yaxis=dict(showgrid=False, side='right'),
        xaxis=dict(showgrid=False, tickformat='%H:%M'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

    # ---------------------------------------------------------
    # 2. LÄNGRE NER: Stapelgrafen (Lik din bild)
    # ---------------------------------------------------------
    st.write("---") # En linje för att separera sektionerna
    st.markdown("#### Timpriser")

    # Förbered data för staplarna (Idag och Imorgon)
    # Skapa en snygg tidssträng "13-14"
    df['Tid_Timme'] = df['Tid'].dt.strftime('%H') + "-" + (df['Tid'] + pd.Timedelta(hours=1)).dt.strftime('%H')
    
    today = nu.date()
    tomorrow = today + pd.Timedelta(days=1)
    
    df_today = df[df['Tid'].dt.date == today].copy()
    df_tomorrow = df[df['Tid'].dt.date == tomorrow].copy()

    # Använd flikar för att spara plats
    tab1, tab2 = st.tabs(["Idag", "Imorgon"])

    with tab1:
        st.caption(f"Priser för {today}")
        if not df_today.empty:
            fig_today = rita_staplar(df_today)
            st.plotly_chart(fig_today, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Ingen data för idag kvar.")

    with tab2:
        st.caption(f"Priser för {tomorrow}")
        if not df_tomorrow.empty:
            fig_tomorrow = rita_staplar(df_tomorrow)
            st.plotly_chart(fig_tomorrow, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Morgondagens priser kommer snart...")
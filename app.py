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
ANTAL_DAGAR = 7

# Gränser för färgläggning
GRANS_BILLIGT = 60
GRANS_DYRT = 100

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

# --- FUNKTION FÖR ATT RITA KALENDER (HEATMAP) ---
def rita_kalender(df):
    # Skapa etiketter
    df['Dag_Datum'] = df['Tid'].dt.date
    df['Dag_Etikett'] = df['Tid'].dt.strftime('%a %d/%m').map(
        lambda x: x.replace('Mon', 'Mån').replace('Tue', 'Tis').replace('Wed', 'Ons')
                   .replace('Thu', 'Tor').replace('Fri', 'Fre').replace('Sat', 'Lör').replace('Sun', 'Sön')
    )
    
    # Pivotera data: Rader=Dagar, Kolumner=Timmar
    dagar_unika = sorted(df['Dag_Datum'].unique())
    
    z_values = []      # Färgvärde (Pris)
    text_values = []   # Text i rutan
    y_labels = []      # Etiketter Y-axel
    
    # För att styra färgen exakt (Grön/Gul/Röd) skapar vi en egen färgskala
    # Men Heatmap i Plotly funkar bäst med numeriska värden. 
    # Vi använder priset direkt.
    
    for dag in dagar_unika:
        dag_data = df[df['Dag_Datum'] == dag]
        # Ta bara med dagar som har hyfsat komplett data (minst 20h)
        if len(dag_data) >= 20:
            y_labels.append(dag_data['Dag_Etikett'].iloc[0])
            priser = dag_data['Pris_Sek'].round(0).astype(int)
            z_values.append(priser.tolist())
            text_values.append(priser.tolist()) # Siffran som syns

    # Vänd ordning så "Idag" hamnar överst
    z_values = z_values[::-1]
    text_values = text_values[::-1]
    y_labels = y_labels[::-1]

    # Skräddarsydd färgskala
    # Vi vill ha Grön upp till 60, Gul upp till 100, Röd över 100.
    # Plotly colorscale är 0.0 till 1.0. Vi får mappa detta.
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=[f"{i:02}" for i in range(24)], # 00, 01...
        y=y_labels,
        text=text_values,
        texttemplate="%{text}", # Visa priset
        textfont={"size": 11, "color": "white"},
        xgap=2, ygap=2, # Mellanrum mellan rutor
        colorscale=[
            [0.0, "#00cc96"], # Grön (Låg)
            [0.4, "#00cc96"], # Grön upp till ca 40% av maxpriset (justerbart)
            [0.40001, "#fecb52"], # Gul start
            [0.7, "#fecb52"], # Gul slut
            [0.70001, "#ef553b"], # Röd start
            [1.0, "#ef553b"]  # Röd slut
        ],
        zmin=0, zmax=150, # Sätter fast skala 0-150 öre
        showscale=False # Dölj färgbalken på sidan
    ))

    fig.update_layout(
        title="7-dygnskalender",
        height=len(y_labels) * 40 + 60, # Anpassa höjd
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(side="top", showgrid=False, title=None, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, title=None),
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

    # 1. MINIMALISTISK TREND (ÖVERST)
    nu = pd.Timestamp.now(tz='Europe/Stockholm')
    nu_timme = nu.floor('h')
    slut_trend = nu + pd.Timedelta(hours=48)
    df_trend = df[(df['Tid'] >= nu_timme) & (df['Tid'] <= slut_trend)].copy()

    aktuellt_pris = df_trend.iloc[0]['Pris_Sek'] if len(df_trend) > 0 else 0
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Just nu", value=f"{aktuellt_pris:.0f} öre")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_trend['Tid'], y=df_trend['Pris_Sek'],
        mode='lines', line=dict(color='#29b6f6',
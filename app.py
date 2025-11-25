import streamlit as st
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import joblib
import holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. INSTÄLLNINGAR ---
st.set_page_config(page_title="Elpris SE4", page_icon="⚡")

LAT_SE = 55.605
LON_SE = 13.003
LAT_DE = 53.551
LON_DE = 9.993
VALUTAKURS = 11.60 
ANTAL_DAGAR = 7

# --- 2. HÄMTA VÄDER ---
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
    return pd.merge(df1, pd.DataFrame(d2), on='Tid')

def ladda_modell():
    try:
        return joblib.load('min_el_modell.pkl')
    except:
        return None

# --- 3. RITA KALENDER (HEATMAP) ---
def rita_kalender(df):
    df['Dag_Datum'] = df['Tid'].dt.date
    df['Dag_Etikett'] = df['Tid'].dt.strftime('%a %d/%m').map(
        lambda x: x.replace('Mon', 'Mån').replace('Tue', 'Tis').replace('Wed', 'Ons')
                   .replace('Thu', 'Tor').replace('Fri', 'Fre').replace('Sat', 'Lör').replace('Sun', 'Sön')
    )
    
    df['Timme_Int'] = df['Tid'].dt.hour
    dagar_unika = sorted(df['Dag_Datum'].unique())
    
    z_values = []      
    text_values = []   
    y_labels = []      
    
    for dag in dagar_unika:
        dag_data = df[df['Dag_Datum'] == dag]
        rad_z = [None] * 24
        rad_text = [""] * 24
        
        for _, row in dag_data.iterrows():
            t = row['Timme_Int']
            if 0 <= t < 24:
                pris = int(round(row['Pris_Sek']))
                rad_z[t] = pris
                rad_text[t] = str(pris)
        
        if any(v is not None for v in rad_z):
            y_labels.append(dag_data['Dag_Etikett'].iloc[0])
            z_values.append(rad_z)
            text_values.append(rad_text)

    # Vänd ordning
    z_values = z_values[::-1]
    text_values = text_values[::-1]
    y_labels = y_labels[::-1]

    if not z_values:
        return None

    # FÄRGSKALA: 0-60 (Grön), 60-100 (Gul), >100 (Röd)
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=[f"{i:02}" for i in range(24)],
        y=y_labels,
        text=text_values,
        texttemplate="%{text}", 
        textfont={"size": 11, "color": "white"},
        xgap=2, ygap=2,
        colorscale=[
            [0.0, "#00cc96"], 
            [0.4, "#00cc96"], 
            [0.40001, "#fecb52"], 
            [0.66, "#fecb52"], 
            [0.66001, "#ef553b"], 
            [1.0, "#ef553b"] 
        ],
        zmin=0, zmax=150, 
        showscale=False,
        hoverongaps=False
    ))

    fig.update_layout(
        title="7-dygnskalender",
        height=len(y_labels) * 40 + 60,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(side="top", showgrid=False, title=None, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, title=None),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
# --- 4. HUVUDPROGRAM ---
st.markdown("### ⚡ Elpris SE4")

model = ladda_modell()

if model is None:
    st.error("⚠️ Ladda upp modellfilen (min_el_modell.pkl)!")
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
        x=df_trend['Tid'], 
        y=df_trend['Pris_Sek'],
        mode='lines', 
        line=dict(color='#29b6f6', width=2, shape='hv'),
        fill='tozeroy', 
        fillcolor='rgba(41, 182, 246, 0.1)'
    ))
    fig_line.update_layout(
        template='plotly_dark', height=200, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, yaxis=dict(showgrid=False, side='right'),
        xaxis=dict(showgrid=False, tickformat='%H:%M'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

    # 2. KALENDER (NEDERST)
    st.write("---")
    
    # Filter: Visa Idag + 6 dagar framåt
    dag_start = nu.normalize()
    dag_slut = dag_start + pd.Timedelta(days=ANTAL_DAGAR)
    df_cal = df[(df['Tid'] >= dag_start) & (df['Tid'] < dag_slut)].copy()
    
    fig_cal = rita_kalender(df_cal)
    if fig_cal:
        st.plotly_chart(fig_cal, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Hämtar data för kalendern...")
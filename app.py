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

st.markdown("""
<style>
:root {
    --green: #0a7d45;
    --yellow: #c58b00;
    --red: #b22222;
    --card: #0f1629;
    --stroke: rgba(255, 255, 255, 0.08);
}
[data-testid="stAppViewContainer"]{
    background: radial-gradient(circle at 20% 20%, #0f172a, #0b1224 45%, #0a0f1f 80%);
    color: #e5e7eb;
}
.hero{
    padding:18px 22px;
    border-radius:18px;
    background: linear-gradient(120deg,#102347,#0b1936,#081226);
    border:1px solid var(--stroke);
    color:#e5e7eb;
    margin-bottom:10px;
}
.hero-title{font-size:22px;font-weight:700;margin-bottom:2px;}
.hero-sub{color:#cbd5e1;font-size:14px;margin-top:4px;}
.metric-row{display:flex;gap:12px;margin:12px 0;}
.metric-card{
    flex:1;
    padding:12px 14px;
    border-radius:14px;
    border:1px solid var(--stroke);
    background:var(--card);
}
.metric-label{
    color:#94a3b8;
    font-size:12px;
    text-transform:uppercase;
    letter-spacing:.04em;
}
.metric-value{font-size:24px;font-weight:700;color:#e2e8f0;}
.metric-band-low{box-shadow:0 0 0 1px rgba(10,125,69,0.3);background:rgba(10,125,69,0.12);}
.metric-band-mid{box-shadow:0 0 0 1px rgba(197,139,0,0.35);background:rgba(197,139,0,0.12);}
.metric-band-high{box-shadow:0 0 0 1px rgba(178,34,34,0.35);background:rgba(178,34,34,0.12);}
</style>
""", unsafe_allow_html=True)

LAT_SE = 55.605
LON_SE = 13.003
LAT_DE = 53.551
LON_DE = 9.993
VALUTAKURS = 11.60
ANTAL_DAGAR = 7
BILLIGT_GRANS = 60
MEDEL_GRANS = 100


def _prisbandklass(v):
    if v < BILLIGT_GRANS:
        return "metric-band-low"
    if v <= MEDEL_GRANS:
        return "metric-band-mid"
    return "metric-band-high"


# --- 2. HÄMTA VÄDER ---
@st.cache_data
def hamta_vader_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

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
            start=pd.to_datetime(r1.Time(), unit="s", utc=True),
            end=pd.to_datetime(r1.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=r1.Interval()),
            inclusive="left"
        ).tz_convert('Europe/Stockholm'),
        "Temp_SE": r1.Variables(0).ValuesAsNumpy(),
        "Vind_SE": r1.Variables(1).ValuesAsNumpy() / 3.6  # API ger km/h, konvertera till m/s
    }
    df1 = pd.DataFrame(d1)

    # Hamburg
    r2 = responses[1].Hourly()
    d2 = {
        "Tid": df1['Tid'],
        "Vind_DE": r2.Variables(1).ValuesAsNumpy() / 3.6  # km/h -> m/s
    }
    return pd.merge(df1, pd.DataFrame(d2), on='Tid')


def ladda_modell():
    try:
        return joblib.load('min_el_modell.pkl')
    except:
        return None


# --- 3. RITA KALENDER (HEATMAP) ---
def skapa_dagforklaringar(df):
    if df.empty:
        return {}

    dagtexter = {}
    for dag, grp in df.groupby(df['Tid'].dt.date):
        avg_pris = grp['Pris_Sek'].mean()
        temp = grp['Temp_SE'].mean()
        vind_se = grp['Vind_SE'].mean()
        vind_de = grp['Vind_DE'].mean()
        roddag = grp['RodDag'].max() > 0
        veckodag = grp['Tid'].dt.dayofweek.iloc[0]

        if avg_pris < BILLIGT_GRANS:
            band = f"Billigt (~{avg_pris:.0f} öre)"
        elif avg_pris <= MEDEL_GRANS:
            band = f"Medel (~{avg_pris:.0f} öre)"
        else:
            band = f"Dyrt (~{avg_pris:.0f} öre)"

        vind_max = max(vind_se, vind_de)
        if vind_max >= 25:
            vind_del = f"mycket kraftig vind (risk för avstängd vindkraft) – SE {vind_se:.1f} m/s, DE {vind_de:.1f} m/s"
        elif vind_max < 3.5:
            vind_del = f"lugn vind (SE {vind_se:.1f} m/s, DE {vind_de:.1f} m/s)"
        elif vind_max <= 8:
            vind_del = f"måttlig vind (SE {vind_se:.1f} m/s, DE {vind_de:.1f} m/s)"
        else:
            vind_del = f"frisk vind (SE {vind_se:.1f} m/s, DE {vind_de:.1f} m/s)"

        if temp <= 0:
            temp_del = f"kallt (~{temp:.0f}°C)"
        elif temp >= 15:
            temp_del = f"milt (~{temp:.0f}°C)"
        else:
            temp_del = f"normalt (~{temp:.0f}°C)"

        helg_del = "helg/helgdag" if (veckodag >= 5 or roddag) else ""

        delar = [d for d in [vind_del, temp_del, helg_del] if d]
        dagtexter[dag] = f"{band} – {', '.join(delar)}"
    return dagtexter


def rita_kalender(df, dagforklaringar=None):
    if df.empty:
        return None

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

    z_values = z_values[::-1]
    text_values = text_values[::-1]
    y_labels = y_labels[::-1]

    if not z_values:
        return None

    zmax = max(160, max(max(row) for row in z_values if row))
    colorscale = [
        [0.0, "#0a7d45"],
        [BILLIGT_GRANS / zmax, "#0a7d45"],
        [(BILLIGT_GRANS / zmax) + 0.0001, "#f0c419"],
        [MEDEL_GRANS / zmax, "#f0c419"],
        [(MEDEL_GRANS / zmax) + 0.0001, "#e25353"],
        [1.0, "#e25353"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=[f"{i:02}" for i in range(24)],
        y=y_labels,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        xgap=2, ygap=2,
        colorscale=colorscale,
        zmin=0, zmax=zmax,
        showscale=False,
        hoverongaps=False,
        hovertemplate="Tid %{x}:00<br>Pris %{z:.0f} öre<extra></extra>"
    ))

    fig.update_layout(
        title="7-dygnskalender (öre/kWh)",
        height=len(y_labels) * 40 + 60,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(side="top", showgrid=False, title=None, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, title=None),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor="#0f172a", font_size=11)
    )
    return fig


# --- 4. HUVUDPROGRAM ---
st.markdown("""
<div class="hero">
  <div class="hero-title">Elpris SE4</div>
  <div class="hero-sub">AI-prognos med prisband för billigt / medel / dyrt</div>
</div>
""", unsafe_allow_html=True)

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

    # 1. TREND + KPI
    nu = pd.Timestamp.now(tz='Europe/Stockholm')
    nu_timme = nu.floor('h')
    slut_trend = nu + pd.Timedelta(hours=48)
    df_trend = df[(df['Tid'] >= nu_timme) & (df['Tid'] <= slut_trend)].copy()

    aktuellt_pris = df_trend.iloc[0]['Pris_Sek'] if len(df_trend) > 0 else 0
    pris_min = df_trend['Pris_Sek'].min() if len(df_trend) > 0 else 0
    pris_max = df_trend['Pris_Sek'].max() if len(df_trend) > 0 else 0
    pris_mean = df_trend['Pris_Sek'].mean() if len(df_trend) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card {_prisbandklass(aktuellt_pris)}'><div class='metric-label'>Just nu</div><div class='metric-value'>{aktuellt_pris:.0f} öre</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card {_prisbandklass(pris_min)}'><div class='metric-label'>Lägsta 48h</div><div class='metric-value'>{pris_min:.0f} öre</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card {_prisbandklass(pris_mean)}'><div class='metric-label'>Snitt 48h</div><div class='metric-value'>{pris_mean:.0f} öre</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card {_prisbandklass(pris_max)}'><div class='metric-label'>Högsta 48h</div><div class='metric-value'>{pris_max:.0f} öre</div></div>", unsafe_allow_html=True)

    trend_start = df_trend['Tid'].min() if len(df_trend) > 0 else nu_timme
    trend_end = df_trend['Tid'].max() if len(df_trend) > 0 else slut_trend
    ymax = max(pris_max, MEDEL_GRANS + 40)

    fig_line = go.Figure()
    fig_line.add_hrect(y0=0, y1=BILLIGT_GRANS, line_width=0, fillcolor='rgba(10,125,69,0.08)', layer="below")
    fig_line.add_hrect(y0=BILLIGT_GRANS, y1=MEDEL_GRANS, line_width=0, fillcolor='rgba(197,139,0,0.10)', layer="below")
    fig_line.add_hrect(y0=MEDEL_GRANS, y1=ymax+40, line_width=0, fillcolor='rgba(178,34,34,0.08)', layer="below")

    fig_line.add_trace(go.Scatter(
        x=df_trend['Tid'],
        y=df_trend['Pris_Sek'],
        mode='lines',
        line=dict(color='#29b6f6', width=2, shape='hv'),
        fill='tozeroy',
        fillcolor='rgba(41, 182, 246, 0.12)',
        hovertemplate="Tid: %{x|%a %H:%M}<br>Pris: %{y:.0f} öre/kWh<extra></extra>"
    ))
    fig_line.add_trace(go.Scatter(
        x=[nu],
        y=[aktuellt_pris],
        mode='markers',
        marker=dict(size=10, color='#29b6f6', line=dict(width=2, color='white')),
        hoverinfo='skip',
        showlegend=False
    ))
    fig_line.update_layout(
        template='plotly_dark', height=260, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        yaxis=dict(showgrid=False, side='right', range=[0, ymax]),
        xaxis=dict(showgrid=False, tickformat='%a %H:%M', range=[trend_start, trend_end]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

    # 2. KALENDER (NEDERST)
    st.write("---")

    dag_start = nu.normalize()
    dag_slut = dag_start + pd.Timedelta(days=ANTAL_DAGAR)
    df_cal = df[(df['Tid'] >= dag_start) & (df['Tid'] < dag_slut)].copy()

    dagforklaringar = skapa_dagforklaringar(df_cal)
    fig_cal = rita_kalender(df_cal, dagforklaringar)
    if fig_cal:
        st.plotly_chart(fig_cal, use_container_width=True, config={'displayModeBar': False})
        if dagforklaringar:
            st.markdown("#### Varför prisnivån per dag")
            for dag in sorted(dagforklaringar.keys()):
                etikett = pd.to_datetime(dag).strftime('%a %d/%m').replace('Mon', 'Mån').replace('Tue', 'Tis').replace('Wed', 'Ons').replace('Thu', 'Tor').replace('Fri', 'Fre').replace('Sat', 'Lör').replace('Sun', 'Sön')
                st.markdown(f"- **{etikett}**: {dagforklaringar[dag]}")
    else:
        st.info("Hämtar data för kalendern...")

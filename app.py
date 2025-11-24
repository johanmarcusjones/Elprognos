import streamlit as st
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import joblib
import holidays
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# --- INSTÄLLNINGAR ---
st.set_page_config(page_title="Elprognos SE4", page_icon="⚡")

LAT_SE = 55.605
LON_SE = 13.003
LAT_DE = 53.551
LON_DE = 9.993
VALUTAKURS = 11.60 
ANTAL_DAGAR = 7

# --- CACHAD DATAHÄMTNING (Så sidan blir snabb) ---
@st.cache_data
def hamta_vader():
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": [LAT_SE, LAT_DE],
        "longitude": [LON_SE, LON_DE],
        "hourly": ["temperature_2m", "wind_speed_10m"],
        "timezone": "Europe/Berlin",
        "forecast_days": ANTAL_DAGAR + 1,
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
    df2 = pd.DataFrame(d2)

    return pd.merge(df1, df2, on='Tid')

def ladda_modell():
    try:
        # Försök ladda modellen
        model = joblib.load('min_el_modell.pkl')
        return model
    except:
        return None

# --- HUVUDPROGRAMMET ---
st.title("⚡ Elprognos SE4 (Malmö)")
st.write("AI-driven prognos baserad på väder i Skåne och Tyskland.")

model = ladda_modell()

if model is None:
    st.error("Kunde inte hitta modell-filen (min_el_modell.pkl). Ladda upp den till GitHub!")
else:
    with st.spinner('Hämtar färskt väder...'):
        df = hamta_vader()

    # Förbered AI-data
    df['Timme'] = df['Tid'].dt.hour
    df['Veckodag'] = df['Tid'].dt.dayofweek
    svenska_helgdagar = holidays.SE()
    df['RodDag'] = df['Tid'].apply(lambda x: 1 if x in svenska_helgdagar else 0)

    # Prognos
    X = df[['Temp_SE', 'Vind_SE', 'Vind_DE', 'Timme', 'Veckodag', 'RodDag']]
    pris_eur = model.predict(X)
    df['Pris_Sek'] = ((pris_eur * VALUTAKURS) / 10) * 1.25

    # Filter
    nu = pd.Timestamp.now(tz='Europe/Stockholm').floor('h')
    slut = nu + pd.Timedelta(days=ANTAL_DAGAR)
    df_plot = df[(df['Tid'] >= nu) & (df['Tid'] <= slut)].copy()

    # --- RITA GRAFEN ---
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_plot['Tid'], df_plot['Pris_Sek'], color='black', linewidth=1.5, zorder=10)
    ax.scatter(df_plot['Tid'], df_plot['Pris_Sek'], s=10, color='black', zorder=11)

    # Färger
    farg_natt = '#e8ebf7'
    farg_fm   = '#fff9c4'
    farg_em   = '#ffe0b2'
    farg_kvall= '#e1bee7'

    start_dag = nu.replace(hour=0, minute=0, second=0, microsecond=0)
    ymax = df_plot['Pris_Sek'].max() * 1.1

    for i in range(ANTAL_DAGAR + 2):
        dag = start_dag + pd.Timedelta(days=i)
        ax.axvspan(dag, dag + pd.Timedelta(hours=6), facecolor=farg_natt, alpha=0.6)
        ax.axvspan(dag + pd.Timedelta(hours=6), dag + pd.Timedelta(hours=12), facecolor=farg_fm, alpha=0.6)
        ax.axvspan(dag + pd.Timedelta(hours=12), dag + pd.Timedelta(hours=18), facecolor=farg_em, alpha=0.6)
        ax.axvspan(dag + pd.Timedelta(hours=18), dag + pd.Timedelta(hours=24), facecolor=farg_kvall, alpha=0.6)
        ax.axvline(x=dag, color='gray', linestyle='-', linewidth=1)

    # Dagar-etiketter
    unika_dagar = df_plot['Tid'].dt.date.unique()
    for d in unika_dagar:
        mitt = pd.Timestamp(d).tz_localize('Europe/Stockholm') + pd.Timedelta(hours=12)
        if mitt >= nu and mitt <= slut:
            t = mitt.strftime('%A\n%d/%m')
            ax.text(mitt, ymax*0.95, t, ha='center', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    ax.set_ylim(0, ymax)
    ax.set_xlim(nu, slut)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    
    # VISA I APPEN
    st.pyplot(fig)
    
    st.subheader("Bästa laddtiderna (Topp 3)")
    # Hitta lägsta priserna
    basta = df_plot.nsmallest(3, 'Pris_Sek')
    for _, row in basta.iterrows():
        st.success(f"🚗 {row['Tid'].strftime('%A %H:%M')} — {row['Pris_Sek']:.1f} öre")
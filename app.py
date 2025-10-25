import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import unicodedata
from pathlib import Path
import json
import re

# --- Configuration & Metadata ---
AUTHOR_NAME = "Lisa BAIL"
AUTHOR_INFO = "ING2-BIOINF, EFREI Paris"

st.set_page_config(
    page_title="Cultural Resilience of French Museums (2014–2023)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------- Utility Functions -------------

def _norm_ascii_lower(s: str) -> str:
    """Converts string to lowercase ASCII for sorting and matching."""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii').lower().strip()

def _norm_name(nom):
    """
    Normalizes region names for reliable GeoJSON/CSV merging (removing accents, 
    apostrophes, and replacing special chars with single hyphens).
    """
    if pd.isna(nom):
        return ""
    # 1. Remove accents
    nom = unicodedata.normalize('NFKD', nom).encode('ASCII', 'ignore').decode('utf-8')
    # 2. Convert to lowercase and simplify spaces/special characters
    nom = re.sub(r"[^a-z0-9]+", "-", nom.lower()).strip("-")
    
    # Specific GeoJSON/CSV consistency fixes (may require adjustment based on your files)
    nom = nom.replace("provence-alpes-cote-d-azur", "provence-alpes-cote-d-azur") 
    nom = nom.replace("ile-de-france", "ile-de-france") 
    
    return nom

# --- Data Loading and Cleaning (with cache) ---
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file_path: str) -> pd.DataFrame:
    """Loads, cleans, and validates museum attendance data."""
    # Load data (CSV separator ';')
    df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)

    # Convert key columns to appropriate types
    if 'annee' in df.columns:
        df['annee'] = pd.to_numeric(df['annee'], errors='coerce').astype('Int64')

    for c in ['region', 'nom_du_musee', 'departement', 'ville']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Safe conversion for numerical attendance columns (handling potential strings/NaN)
    cols_numeriques = [
        'payant', 'gratuit', 'total',
        'individuel', 'scolaires', 'groupes_hors_scolaires',
        'moins_18_ans_hors_scolaires', '_18_25_ans'
    ]
    for col in cols_numeriques:
        if col in df.columns:
            # Replaces blanks/non-numeric with NaN, then converts to Int64 (Pandas integer with NaN support)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('Int64')

    # Consistency check: total ~ payant + gratuit (with tolerance for small reporting errors)
    if {'payant', 'gratuit', 'total'}.issubset(df.columns):
        calc = df['payant'].fillna(0) + df['gratuit'].fillna(0)
        # Fix 'total' if missing or significantly different (> 1000) from the sum
        mask_fix = df['total'].isna() | (df['total'] - calc).abs() > 1000
        df.loc[mask_fix, 'total'] = calc.loc[mask_fix]

    # Keep only useful rows (total attendance > 0)
    if 'total' in df.columns:
        df = df[df['total'].fillna(0) > 0].copy()

    return df

# --- Data Loading ---
DATA_PATH = 'data/musees.csv'
try:
    data = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Error: Data file '{DATA_PATH}' not found. Please ensure 'musees.csv' is in the 'data' folder.")
    st.stop()
except Exception as e:
    st.error(f"Error during data loading or cleaning: {e}")
    st.stop()

# --- Sidebar (Filters) ---
st.sidebar.title("Analysis Parameters")
# --- Metric Selector (used in multiple visualizations) ---
metric = st.sidebar.selectbox(
    "Select metric to analyze:",
    ["total", "payant", "gratuit"],
    index=0
)
st.sidebar.caption("Choose between total entries, paying visitors, or free visitors.")


# Region Selector
regions_available = sorted(
    data['region'].dropna().astype(str).unique().tolist(),
    key=_norm_ascii_lower
) if 'region' in data.columns else []

region_selection = st.sidebar.multiselect(
    "Filter by Region",
    regions_available,
    # KEY CHANGE: Empty list by default for global view
    default=[] 
)

# Filter DataFrame by Region
df_filtered = data[data['region'].isin(region_selection)].copy() if region_selection else data.copy()

# Museum Selector (limited to selected regions)
museums_available = sorted(
    df_filtered['nom_du_musee'].dropna().astype(str).unique().tolist(),
    key=_norm_ascii_lower
) if 'nom_du_musee' in df_filtered.columns else []

museum_selection = st.sidebar.multiselect(
    "Filter by Museum (in selected Region)",
    museums_available,
    default=[]
)

# Final filtered DataFrame
df_final = df_filtered[df_filtered['nom_du_musee'].isin(museum_selection)].copy() if museum_selection else df_filtered.copy()

# Stop if no data matches filters
if df_final.empty:
    st.error("No data to display. Please adjust your Region and/or Museum filters.")
    st.stop()

# --- Dashboard Title and Signature ---
st.title("🇫🇷 Cultural Resilience of French Museums (2014–2023)")
st.markdown(f"**Data Storytelling by {AUTHOR_NAME}, {AUTHOR_INFO}**")
st.caption(
    "Source: data.gouv.fr — « Fréquentation des musées de France ». "
    "Method: Annual aggregation, total corrected (paid+free) for consistency. "
    "Limitations: Significant data gaps in age categories, heterogeneous reporting among museums."
)
st.markdown("---")
st.info("**Has museum attendance in France fully recovered since 2019? Which regions and audiences are driving the rebound?**")
st.caption("Target audience: cultural policy makers and museum managers who want to understand post-COVID dynamics.")



# --- SECTION 1: CONTEXT & SHOCK IMPACT (Problem) ---
st.header("1. Context: The 2020 Shock and the Recovery Trajectory")

# KPI Calculation (driven by `metric`)
df_metric_yearly_kpi = df_final.groupby('annee', as_index=False)[metric].sum()
v2019 = float(df_metric_yearly_kpi.loc[df_metric_yearly_kpi['annee'].eq(2019), metric].sum())
v2020 = float(df_metric_yearly_kpi.loc[df_metric_yearly_kpi['annee'].eq(2020), metric].sum())
v2023 = float(df_metric_yearly_kpi.loc[df_metric_yearly_kpi['annee'].eq(2023), metric].sum())

shock_2020 = (100 * (v2020 - v2019) / v2019) if v2019 > 0 else 0.0
recovery_2023 = (100 * (v2023 - v2019) / v2019) if v2019 > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric(f"{metric.capitalize()} (2023)", f"{v2023:,.0f}".replace(',', ' '), delta=f"{recovery_2023:.1f}% vs 2019")
c2.metric("Reference (2019)", f"{v2019:,.0f}".replace(',', ' '), help="Value before the COVID-19 crisis.")
c3.metric("Max. Shock (2020 vs 2019)", f"{shock_2020:.1f}%", delta_color="inverse",
          help="The sharpest decline due to the crisis.")

# --- Narrative: automatic KPI insight (EN) ---
y_pre = df_metric_yearly_kpi[df_metric_yearly_kpi['annee'].between(2014, 2019)]
if not y_pre.empty and (y_pre['annee'] == 2014).any() and (y_pre['annee'] == 2019).any():
    start_2014 = float(y_pre.loc[y_pre['annee'].eq(2014), metric].sum()) or 0.0
    end_2019 = float(y_pre.loc[y_pre['annee'].eq(2019), metric].sum()) or 0.0
    pre_trend = ((end_2019 / start_2014) - 1) * 100 if start_2014 > 0 else np.nan
else:
    pre_trend = np.nan

pre_trend_txt = f"{pre_trend:.1f}%" if pd.notna(pre_trend) else "N/A"
st.markdown(f"""
<div style="background:#0077b6;padding:12px;border-radius:8px;">
<b>Quick read ({metric}):</b> between 2014 and 2019, the metric grew by <b>{pre_trend_txt}</b>. 
The 2020 shock was <b>{shock_2020:.1f}%</b> vs 2019, and the 2023 level stands <b>{recovery_2023:.1f}%</b> compared with 2019.
This frames the recovery trajectory you see in the chart below.
</div>
""", unsafe_allow_html=True)


# --- MAIN TIME SERIES (driven by the selected `metric`) ---
st.markdown(f"### Annual Evolution of {metric.capitalize()} Entries (2014–2023)")

# Agrège la métrique choisie
df_metric_yearly = (
    df_final.groupby('annee', as_index=False)[metric]
            .sum()
            .sort_values('annee')
)

fig_line = px.line(
    df_metric_yearly,
    x='annee', y=metric, markers=True,
    title=f"Evolution of {metric.capitalize()} Attendance (2014–2023)",
    labels={'annee': 'Year', metric: 'Entries'},
    text=metric
)
fig_line.update_traces(
    textposition="bottom right",
    hovertemplate="Year=%{x}<br>Entries=%{y:,.0f}<extra></extra>"
)

# Ligne de base = valeur 2019 pour la métrique choisie (si dispo)
baseline_2019 = float(
    df_metric_yearly.loc[df_metric_yearly['annee'].eq(2019), metric].sum()
)
if baseline_2019 > 0:
    fig_line.add_hline(
        y=baseline_2019,
        line_dash="dot",
        annotation_text="2019 baseline",
        annotation_position="bottom right",
        line_color="gray"
    )

# Annotation COVID + format des tick labels
fig_line.add_vrect(x0=2019.5, x1=2020.5, fillcolor="red", opacity=0.08, line_width=0)
max_y = float(df_metric_yearly[metric].max() or 0)
fig_line.add_annotation(x=2020, y=max_y, text="COVID Shock", showarrow=True, arrowhead=2)
fig_line.update_layout(yaxis_tickformat=",.0f")

st.plotly_chart(fig_line, use_container_width=True)


st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; color: #333333;">
    <h4 style="color:#0077b6;">Insight: The 2020 Trough</h4>
    The curve illustrates the unprecedented impact of lockdowns: a <b>brutal drop in attendance in 2020</b>, 
    breaking the pre-crisis growth trend. Museums lost several years of work in a single year. 
    The data shows a clear rebound in 2022 and 2023, but the <b>speed and completeness of the recovery vary significantly</b>.
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# --- SECTION 2: PUBLIC CATEGORIES & ECONOMIC MIX (Analysis) ---
st.header("2. Analysis: Audience Segments and Economic Mix")

# 2.1 Paid vs Free (stacked)
df_paid_free = df_final.groupby('annee', as_index=False)[['payant', 'gratuit']].sum()
fig_bar_paid_free = go.Figure(data=[
    go.Bar(name='Paid Entries', x=df_paid_free['annee'], y=df_paid_free['payant'], marker_color='#1f77b4'),
    go.Bar(name='Free Entries', x=df_paid_free['annee'], y=df_paid_free['gratuit'], marker_color='#ff7f0e')
])
fig_bar_paid_free.update_layout(
    barmode='stack',
    title='Evolution of Paid vs. Free Attendance',
    xaxis_title='Year', yaxis_title="Number of Entries",
    legend_title="Entry Type"
)
st.plotly_chart(fig_bar_paid_free, use_container_width=True)

st.markdown("""
<div style="background-color: #e0f2f1; padding: 10px; border-radius: 5px; color: #004d40;">
    <b>Key Takeaway:</b> The Paid vs. Free analysis reveals the economic model's resilience. 
    A strong rebound in <b>Paid Entries</b> (often international tourists, single visitors) supports museum revenues. 
    A higher proportion of <b>Free Entries</b> might indicate successful local/educational outreach policies, but can signal ongoing economic pressure.
</div>
""", unsafe_allow_html=True)

# 2.2 2019 vs 2023 Breakdown (100% stack)
cat_cols = ['individuel', 'scolaires', 'groupes_hors_scolaires', 'moins_18_ans_hors_scolaires', '_18_25_ans']
existing_cats = [c for c in cat_cols if c in df_final.columns]
df_cats = df_final.groupby('annee', as_index=False)[existing_cats].sum()
df_cats_2y = df_cats[df_cats['annee'].isin([2019, 2023])].melt('annee', var_name='category', value_name='val')

if not df_cats_2y.empty:
    # Rename categories for better display
    category_map = {
        'individuel': 'Individual',
        'scolaires': 'School Groups',
        'groupes_hors_scolaires': 'Non-School Groups',
        'moins_18_ans_hors_scolaires': 'Under 18 (Non-School)',
        '_18_25_ans': '18-25 yrs'
    }
    df_cats_2y['category'] = df_cats_2y['category'].map(category_map)
    
    # Calculate share of total
    df_cats_2y['share'] = df_cats_2y.groupby('annee')['val'].transform(lambda s: s / s.sum() if s.sum() else 0)
    fig_cats = px.bar(
        df_cats_2y, x='annee', y='share', color='category', barmode='stack',
        title='Public Category Distribution (Share, 2019 vs 2023)',
        labels={'annee':'Year','share':'Share','category':'Category'},
        color_discrete_sequence=px.colors.qualitative.D3
    )
    fig_cats.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig_cats, use_container_width=True)
else:
    st.info("Public category columns are too sparse to compare 2019 vs 2023.")


# --- SECTION 3: MUSEUM-LEVEL ANALYSIS ---
st.header("3. Museum-Level Analysis: Leading and Lagging")

st.markdown("#### Top 20 museums in 2023 and how they compare to 2019")

top_df = df_final.groupby(['nom_du_musee','annee'])[metric].sum().unstack(fill_value=0)
if 2019 in top_df.columns and 2023 in top_df.columns:
    top_df["% vs 2019"] = np.where(top_df[2019]>0, (top_df[2023]/top_df[2019]-1)*100, np.nan)
    top20 = top_df.sort_values(2023, ascending=False).head(20).reset_index()
    fig_top = px.bar(
        top20,
        x='nom_du_musee', y=2023,
        color='% vs 2019',
        color_continuous_scale=px.colors.diverging.RdBu_r,
        title=f"Top 20 museums by {metric.capitalize()} (2023)",
        labels={'nom_du_musee': 'Museum', 2023: metric.capitalize()},
        category_orders={'nom_du_musee': top20['nom_du_musee'].tolist()}
    )
    fig_top.update_layout(xaxis_tickangle=45, height=600)
    st.plotly_chart(fig_top, use_container_width=True)

    # --- Narrative: Top-20 movers (EN) ---
if 2019 in top_df.columns and 2023 in top_df.columns:
    improved = (top_df
                .sort_values("% vs 2019", ascending=False)
                .head(3)
                .index.tolist())
    declined = (top_df
                .sort_values("% vs 2019", ascending=True)
                .head(3)
                .index.tolist())
    st.caption(
        "Most improved since 2019: " + (", ".join(improved) if improved else "N/A") +
        " · Biggest declines: " + (", ".join(declined) if declined else "N/A")
    )


    st.markdown("""
<div style="background-color: #e0f2f1; padding: 10px; border-radius: 5px; color: #004d40;">
    <b>Interpretation:</b> The color of the bar shows the resilience of the museum (Green=growth vs 2019, Red=decline). 
    While absolute attendance is high (Y-axis), museums below the 0% mark (Yellow/Red) are still strategically lagging in their recovery efforts.
</div>
    """, unsafe_allow_html=True)
else:
    st.info("Not enough data for comparison between 2019 and 2023 at the museum level.")



st.markdown("---")


# --- SECTION 4: REGIONAL DISPARITIES (Implications) ---
st.header("4. Regional Disparities: Geographic Focus")

st.markdown("""
<div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; color: #b35900;">
    <h4 style="color: #b35900;">Geographic Focus: Who Recovered Best?</h4>
    The chart and map below highlight <b>major regional disparities</b>. Regions in green (Recovery > 0%) have managed to surpass their 2019 attendance levels. 
    This is often attributed to strong local audience engagement, successful targeted campaigns, or less reliance on specific international tourism. 
    Regions in yellow/red (Recovery < 0%) need specific cultural support strategies to close the gap.
</div>
""", unsafe_allow_html=True)

# --- Regional Data Preparation (using full 'data' for the map - NATIONAL PICTURE) ---
# Correction: Use unfiltered 'data' to show the national picture of disparities, regardless of sidebar filter
df_regional = data.groupby(['region', 'annee'])['total'].sum().unstack(fill_value=0).reset_index()

# Recovery Ratio 2023 vs 2019
if 2019 in df_regional.columns and 2023 in df_regional.columns:
    df_regional['Recovery_Ratio'] = np.where(df_regional[2019] > 0, df_regional[2023] / df_regional[2019], np.nan)
    df_regional['Recovery_%'] = (df_regional['Recovery_Ratio'] - 1) * 100
else:
    df_regional['Recovery_%'] = np.nan

df_regional_display = df_regional[['region'] + [c for c in [2019, 2023] if c in df_regional.columns] + ['Recovery_%']].copy()
df_regional_display = df_regional_display.rename(columns={2019: 'Entries 2019', 2023: 'Entries 2023'})
df_regional_display['Recovery_%'] = df_regional_display['Recovery_%'].map(lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A')
if 'Entries 2023' in df_regional_display.columns:
    df_regional_display = df_regional_display.sort_values(by='Entries 2023', ascending=False)

st.markdown("This table shows the rate of return to pre-crisis attendance (2019) by region. A positive percentage indicates growth.")
st.dataframe(df_regional_display, use_container_width=True)

# Horizontal Bar Chart for Recovery %
df_recovery_chart = df_regional[np.isfinite(df_regional['Recovery_%'])].copy()
df_recovery_chart = df_recovery_chart.sort_values(by='Recovery_%', ascending=True)
fig_recovery = px.bar(
    df_recovery_chart, x='Recovery_%', y='region', orientation='h',
    title='Attendance Recovery Rate (2023 vs 2019) by Region',
    labels={'region':'Region','Recovery_%':'Recovery Rate (%)'},
    color='Recovery_%', 
    color_continuous_scale=px.colors.diverging.RdBu_r
)
st.plotly_chart(fig_recovery, use_container_width=True)

# --- Narrative: regional winners/laggards (EN) ---
if np.isfinite(df_regional['Recovery_%']).any():
    # Safeguards for empty or all-NaN cases
    valid = df_regional[np.isfinite(df_regional['Recovery_%'])].copy()
    if not valid.empty:
        best = valid.iloc[valid['Recovery_%'].argmax()]
        worst = valid.iloc[valid['Recovery_%'].argmin()]
        st.markdown(f"""
**Map takeaway:** The strongest recovery in 2023 vs 2019 appears in **{best['region']} (+{best['Recovery_%']:.1f}%)**,
while the weakest is **{worst['region']} ({worst['Recovery_%']:.1f}%)**.
This gap suggests different dependencies on international tourism and varying strengths in local audience engagement.
""")
else:
    st.caption("Regional recovery insights are unavailable because 2019/2023 totals are missing for some regions.")


st.subheader("Evolution by region (2014–2023)")

st.markdown("Visualizing each region individually reveals different recovery trajectories across the country.")

ts_reg = df_final.groupby(['region','annee'])[metric].sum().reset_index()
fig_small = px.line(
    ts_reg, x='annee', y=metric,
    facet_col='region', facet_col_wrap=4,
    height=800, markers=True,
    labels={'annee':'Year', metric:'Entries'}
)
fig_small.update_traces(line=dict(width=1))
st.plotly_chart(fig_small, use_container_width=True)


st.markdown("""
<div style="background-color: #e0f2f1; padding: 10px; border-radius: 5px; color: #004d40;">
    <b>Contextual Analysis:</b> These small charts (facet plots) are essential for identifying 
    <b>outliers</b> (e.g., a region that did not suffer the 2020 shock as severely, or a region that stagnated after the rebound). 
    They confirm that recovery is not a uniform national trend, but a sum of unique local stories.
    The 2020 trough is visible almost everywhere, but the rebound pace differs.
Watch for regions with early or steeper recoveries and those that remain flat after 2021—these patterns hint at
different audience mixes, product/offer dynamics, and reliance on tourism vs. local demand.
</div>
""", unsafe_allow_html=True)


# --- Choropleth Map (GeoPandas) ---
st.markdown("### Regional Choropleth Map: A Post-Crisis Divide")

REGION_CODE = {
    "auvergne-rhone-alpes": "84","bourgogne-franche-comte": "27","bretagne": "53",
    "centre-val-de-loire": "24","corse": "94","grand-est": "44","hauts-de-france": "32",
    "ile-de-france": "11","normandie": "28","nouvelle-aquitaine": "75","occitanie": "76",
    "pays-de-la-loire": "52","provence-alpes-cote-d-azur": "93", # Added normalization fix here
    "guadeloupe":"01","martinique":"02","guyane":"03","la-reunion":"04","mayotte":"06",
}
geojson_path = Path("data/regions.geojson")

try:
    import geopandas as gpd
    
    if not geojson_path.exists():
        raise FileNotFoundError("GeoJSON file `data/regions.geojson` not found.")

    # Use @st.cache_resource for GeoPandas data
    @st.cache_resource
    def load_regions_geojson(path: Path):
        gdf = gpd.read_file(path)
        return gdf.to_crs(epsg=4326)

    gdf = load_regions_geojson(geojson_path)

    # --- GeoJSON/Data Merge Logic ---
    if "nom" in gdf.columns:
        gdf["key"] = gdf["nom"].apply(_norm_name)
        join_on_name = True
    elif "code_insee" in gdf.columns:
        gdf["key"] = gdf["code_insee"].astype(str).str.zfill(2)
        join_on_name = False
    else:
        raise ValueError("GeoJSON must contain a 'nom' or 'code_insee' field.")

    # Use the regional data calculated from full 'data'
    df_map = df_regional.copy() 
    df_map["region_norm"] = df_map["region"].apply(_norm_name)
    
    # Rename columns for display safety
    if 2019 in df_map.columns: df_map = df_map.rename(columns={2019: "Entries 2019"})
    if 2023 in df_map.columns: df_map = df_map.rename(columns={2023: "Entries 2023"})
    
    # Perform Merge
    if join_on_name:
        merged = gdf.merge(df_map, left_on="key", right_on="region_norm", how="left")
    else:
        df_map["code_insee"] = df_map["region_norm"].map(REGION_CODE) # Create INSEE code for merge
        merged = gdf.merge(df_map, left_on="key", right_on="code_insee", how="left")
    
    # Build geojson from the merged GeoDataFrame
    merged_geojson = json.loads(merged.to_json())

    # Choropleth Map Plotly
    fig_map = px.choropleth_mapbox(
        merged,
        geojson=merged_geojson,
        locations=merged.index.astype(str), # Use index as feature ID
        featureidkey="id", # Set to "id" if GeoJSON features have it
        color="Recovery_%",
        color_continuous_scale=px.colors.diverging.RdBu_r,
        mapbox_style="carto-positron",
        # ADJUSTED FOR WIDER VIEW (France + DOM-TOM)
        zoom=2.8, 
        center={"lat": 44.0, "lon": -1.0},
        opacity=0.75,
        labels={"Recovery_%": "Recovery (%)"},
        hover_name="region",
        hover_data={"Entries 2019": ":,.0f" if "Entries 2019" in merged.columns else False,
                    "Entries 2023": ":,.0f" if "Entries 2023" in merged.columns else False,
                    "Recovery_%": ":.1f"},
        title="Recovery 2023 vs 2019 by Region (Choropleth)",
    )
    # Ensure full interactivity
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

except ModuleNotFoundError:
    st.info("GeoPandas is not installed. Please add necessary libraries to requirements.txt.")
except FileNotFoundError as e:
    st.info(f"{e} Place a GeoJSON file of regions (with a 'nom' or 'code_insee' field) in `data/regions.geojson`.")
except Exception as e:
    st.info(f"Could not display GeoPandas choropleth: {e}")

# --- What-if Scenario ---
st.subheader("What-if ? simulation: impact of pricing policy")

price = st.number_input("Average ticket price (€)", min_value=5.0, max_value=20.0, value=10.0)
delta = st.slider("Change in paying attendance (%)", -50, 50, 0)

rev_now = df_final[df_final['annee'] == 2023]['payant'].sum() * price
rev_new = rev_now * (1 + delta/100)

st.metric("Simulated 2023 revenue", f"{rev_new:,.0f} €".replace(',', ' '), f"{delta:+.0f}% vs actual")
st.caption(
    "This simulation assumes an average ticket price applied to all paying visitors in 2023. "
    "It illustrates the sensitivity of total revenue to price changes."
)

st.markdown("""
<div style="background-color: #e0f2f1; padding: 10px; border-radius: 5px; color: #004d40;">
    The 'What-if' tool is key for decision-makers. It quickly quantifies the
    <b>revenue elasticity</b> of paid entries against changes in attendance (linked to pricing or marketing). 
    A small negative attendance change can lead to a significant revenue drop, emphasizing the importance of pricing stability or free-entry policies.
</div>
""", unsafe_allow_html=True)


# --- Data Quality & Export ---
st.markdown("### Data Quality & Limitations")
dq_cols = [c for c in ['individuel','scolaires','groupes_hors_scolaires','moins_18_ans_hors_scolaires','_18_25_ans'] if c in data.columns]
if dq_cols:
    na_rate = (data[dq_cols].isna().mean().sort_values(ascending=False) * 100).round(1)
    st.write("Missing values rate (main audience categories):")
    st.dataframe(na_rate.to_frame('% Missing'))
else:
    st.write("Detailed audience columns are not available in this file.")

if {'payant','gratuit','total'}.issubset(data.columns):
    total_diff = (data['total'].fillna(0) - (data['payant'].fillna(0) + data['gratuit'].fillna(0))).abs().mean()
    st.caption(f"Average absolute difference |total - (paid+free)| ≈ **{total_diff:.0f}** visitors …")

st.download_button(
    "Download Selection (CSV)",
    data=df_final.to_csv(index=False).encode('utf-8'),
    file_name="selected_museums_data.csv",
    mime="text/csv"
)

st.subheader("Additional Data Quality Checks")

dup = data.duplicated(subset=['nom_du_musee','annee'], keep=False).sum()
st.caption(f"Potential duplicates (same museum × year): {dup}")

cov = (data.groupby('annee')[['payant','gratuit']]
       .apply(lambda d: 100*d.notna().mean())).round(1)
st.write("Coverage (%) of paying/free data per year")
st.dataframe(cov)


st.markdown("## Key Insights and conclusion")

st.success(f"""
**1) National resilience with local asymmetries.**  
By 2023, **{metric}** stands **{recovery_2023:.1f}%** vs 2019. The rebound is real but uneven: 
some territories and museums outperform, others still lag behind pre-COVID benchmarks.

**2) The paid/free mix shapes recovery economics.**  
A healthy rebound of **paid entries** is critical for financial stability. A higher **free** share can reflect 
access policies and local outreach successes, but may indicate pressure on revenue if not matched by funding.

**3) Geography matters.**  
Regions with stronger local engagement, refreshed programming, and lower reliance on long-haul tourism tend to exceed 2019. 
Others remain below baseline—these are prime candidates for targeted support.

**4) Museum-level leaders vs. laggards.**  
Top performers often combine compelling content, strong branding, events, and good accessibility. 
Persistent under-performance is frequently tied to international exposure, narrow audience mixes, or offer fatigue.

**Actionable next steps.**  
- Double down on **local audiences** (families, schools, 18–25) where recovery is incomplete.  
- **Tune pricing and promotions** carefully: the what-if suggests revenue is highly sensitive to paid footfall.  
- **Standardize audience tracking** (age categories are sparse) to measure accessibility and inclusion.  
- Share **best practices** from leading regions/museums (programming, partnerships, proximity marketing).
""")

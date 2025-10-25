# StreamlitApp25_20221720_BAIL_BIO
# Cultural Resilience of French Museums (2014–2023)

## Project Goal: Data Storytelling & Strategic Analysis

This Streamlit dashboard analyzes the resilience of French museums following the major shock of the 2020 health crisis. The objective is to guide cultural policy makers and museum managers by identifying:
1. The **speed and completeness of the national recovery** (2023 vs 2019 baseline).
2. The **economic mix** of paid vs. free attendance driving the rebound.
3. The **major regional disparities** in recovery across the French territory.

The application follows a clear narrative structure: **Problem (Shock) → Analysis (Public Mix) → Implications (Geographic Focus)**.

## Key Insight

By 2023, national attendance is near pre-crisis levels, but this masks a **significant regional and structural divide**. Regions with strong local audiences or specific tourism markets (e.g., [Nom de la région la plus performante]) show growth, while others lag significantly behind (e.g., [Nom de la région la moins performante]), suggesting the need for **targeted support policies**.

## Technical Requirements and Setup

To run this application locally in a clean environment:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/lisacsd/StreamlitApp25_20221720_BAIL_BIO.git
    cd StreamlitApp25_20221720_BAIL_BIO
    ```

2.  **Create and Activate Environment (Recommended):**
    ```bash
    # Using conda
    conda create -n musees-env python=3.10
    conda activate musees-env
    
    # OR Using venv
    python -m venv musees-env
    source musees-env/bin/activate  # on Linux/macOS
    # .\musees-env\Scripts\activate  # on Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    The application will automatically open in your default browser.

## Project Structure

- `app.py`: Main Streamlit application code (data loading, cleaning, visualizations, and narrative).
- `requirements.txt`: List of required Python dependencies (`streamlit`, `pandas`, `plotly`, `geopandas`, etc.).
- `data/musees.csv`: The raw dataset for museum attendance.
- `data/regions.geojson`: GeoJSON file used for the choropleth map visualization.

## Link of the steamlit app : 
https://culturalresilienceoffrenchmuseums.streamlit.app/
## Link of the github repository :
https://github.com/lisacsd/StreamlitApp25_20221720_BAIL_BIO.git 
## Link of the Loom video : 
https://www.loom.com/share/7e076387a618463db7027d7995b16245
## Link of the storyboard :
https://www.canva.com/design/DAG2zfcWmh0/w2prwPHh8ni3c40AuaVDSw/edit?utm_content=DAG2zfcWmh0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 


## Data Source
The dataset used for this analysis is: **Fréquentation des musées de France** from [data.gouv.fr](https://www.data.gouv.fr/datasets/frequentation-des-musees-de-france-1/)

***

## Author
**Lisa BAIL** – ING2-BIOINF, EFREI Paris – Data Visualization Project 2025


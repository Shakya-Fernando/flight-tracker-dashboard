import gradio as gr
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import contextily as ctx
from pathlib import Path
from datetime import datetime

### VARIABLES ###

FLIGHT_LOG_PATH = "../data/raw/flight-logs/"
METADATA_CSV = "../data/raw/metadata/waterways-merged-precinct-metadata-20250625.csv"
FLIGHT_CRS = 'EPSG:4326'

### FUNCTIONS ###

def load_metadata():
    """Load metadata CSV and return filtered dataframe"""
    try:
        if not os.path.exists(METADATA_CSV):
            print(f"Metadata file {METADATA_CSV} not found")
            return pd.DataFrame()
        
        df = pd.read_csv(METADATA_CSV)
        required_columns = ['Site name', 'Precinct', 'Day of the week', 'Hour', 'Time of the day', 'Date', 'Flight Log Filename']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in metadata: {missing_columns}")
            return pd.DataFrame()
        
        metadata_table = df[required_columns].copy()
        metadata_table = metadata_table.drop_duplicates()
        metadata_table = metadata_table.sort_values('Site name')
        
        print(f"Loaded metadata for {len(metadata_table)} sites")
        return metadata_table
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return pd.DataFrame()

metadata_df = load_metadata()

def get_site_names() -> list[str]:
    """Get list of unique site names"""
    return metadata_df['Site name'].unique().tolist()

def get_flight_logs_for_site(site_name: str) -> pd.DataFrame:
    """Get flight logs for selected site"""
    return metadata_df[metadata_df['Site name'] == site_name][['Precinct', 'Flight Log Filename', 'Date', 'Day of the week', 'Hour', 'Time of the day']]

def load_flight_data(csv_file: str) -> pd.DataFrame:
    """Load and clean flight data from CSV"""
    file_path = os.path.join(FLIGHT_LOG_PATH, csv_file)
    df = pd.read_csv(file_path)
    
    if 'isVideo' in df.columns:
        df = df[df['isVideo'] == 1]
    
    coord_map = {
        'latitude': ['latitude'],
        'longitude': ['longitude']
    }
    
    for std_name, alternatives in coord_map.items():
        if std_name not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df[std_name] = df[alt]
                    break
    
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
    
    return df

def create_flight_visualisation(csv_file: str) -> plt.Figure:
    """Create flight path visualisation"""
    if not csv_file:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No CSV file selected', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    try:
        df = load_flight_data(csv_file)
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("No coordinate columns found")
        
        if len(df) == 0:
            raise ValueError("No valid coordinate data")
        
        
        flight_name = Path(csv_file).stem.replace("-Flight-Airdata", "")
        
        plt.close('all')
        
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=FLIGHT_CRS)
        gdf_mercator = gdf.to_crs('EPSG:3857')
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        if len(gdf) > 1:
            coords = [(row.geometry.x, row.geometry.y) for _, row in gdf.iterrows()]
            flight_path = LineString(coords)
            path_gdf = gpd.GeoDataFrame([{'geometry': flight_path}], crs=FLIGHT_CRS)
            path_gdf_mercator = path_gdf.to_crs('EPSG:3857')
            path_gdf_mercator.plot(ax=ax, color='red', linewidth=3, alpha=0.8, label='Flight Path')
        
        colours = plt.cm.viridis(np.linspace(0, 1, len(gdf_mercator)))
        gdf_mercator.plot(ax=ax, color=colours, markersize=30, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        if len(gdf_mercator) > 1:
            gdf_mercator.iloc[0:1].plot(ax=ax, color='green', markersize=150, marker='^', 
                                      label='Start', edgecolor='white', linewidth=2)
            gdf_mercator.iloc[-1:].plot(ax=ax, color='red', markersize=150, marker='v', 
                                       label='End', edgecolor='white', linewidth=2)
        
        bounds = gdf_mercator.total_bounds
        centre_x, centre_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        
        buffer = 0.1
        width_buffered = width * (1 + buffer)
        height_buffered = height * (1 + buffer)
        
        fig_aspect = 4/3 
        if width_buffered / height_buffered > fig_aspect:
            final_width = width_buffered
            final_height = width_buffered / fig_aspect
        else:
            final_height = height_buffered
            final_width = height_buffered * fig_aspect
        
        ax.set_xlim(centre_x - final_width/2, centre_x + final_width/2)
        ax.set_ylim(centre_y - final_height/2, centre_y + final_height/2)
        
        ctx.add_basemap(ax, crs=gdf_mercator.crs.to_string(), 
                       source=ctx.providers.OpenStreetMap.Mapnik, alpha=1)
        
        title = f'Flight Path: {flight_name}'       
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_axis_off()
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating visualisation: {e}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

def update_flight_logs_and_plot(site_name: str):
    flight_logs = get_flight_logs_for_site(site_name)
    if flight_logs.empty:
        return flight_logs, None
    csv_file = flight_logs.iloc[0]['Flight Log Filename']
    plot = create_flight_visualisation(csv_file)
    return flight_logs, plot

def update_map(flight_logs, selection: gr.SelectData):
    """Update map according to selected row"""
    if selection is not None:
        csv_files = flight_logs.loc[selection.index, 'Flight Log Filename']
        return create_flight_visualisation(csv_files.iloc[0])  
    else:
        return None

### GRADIO INTERFACE ###

with gr.Blocks() as demo:
    gr.Markdown("# Flight Path Visualiser")
    gr.Markdown(f"CSV directory: `{FLIGHT_LOG_PATH}`")
    gr.Markdown(f"Metadata file: `{METADATA_CSV}`")
    
    with gr.Row():
        metadata_table = gr.Dataframe(
            label="Site Metadata",
            headers=["Site name", "Precinct", "Day of the week", "Hour", "Time of the day", "Date"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            value=metadata_df[["Site name", "Precinct", "Day of the week", "Hour", "Time of the day", "Date"]].drop_duplicates()
        )

    site_names = get_site_names()
    site_dropdown = gr.Dropdown(label="Select Site", choices=site_names)  
        
    with gr.Row():
        flight_logs_table = gr.Dataframe(
            label="Flight Logs",
            headers=["Precinct", "Flight Log Filename", "Date", "Day of the week", "Hour", "Time of the day"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )
    with gr.Row():    
        flight_plot = gr.Plot(label="Flight Path Visualisation")

    site_dropdown.change(update_flight_logs_and_plot, inputs=site_dropdown, outputs=[flight_logs_table, flight_plot])

    flight_logs_table.select(update_map, inputs=[flight_logs_table], outputs=flight_plot)

    demo.load(lambda: update_flight_logs_and_plot(site_names[0]), outputs=[flight_logs_table, flight_plot])

if __name__ == "__main__":
    demo.launch()
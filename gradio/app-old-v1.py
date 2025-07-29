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
    """Load metadata CSV and return filtered dataframe with value counts"""
    try:
        if not os.path.exists(METADATA_CSV):
            print(f"Metadata file {METADATA_CSV} not found")
            return pd.DataFrame()
        
        df = pd.read_csv(METADATA_CSV)
        required_columns = ['Site name', 'Day of the week', 'Time of the day', 'Flight Log Filename']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in metadata: {missing_columns}")
            return pd.DataFrame()
        
        # Group by Site name and get value counts for each categorical column
        site_summary = []
        
        for site in df['Site name'].unique():
            site_data = df[df['Site name'] == site]
            
            # Count Day of the week categories
            day_counts = site_data['Day of the week'].value_counts()
            weekday_count = day_counts.get('Weekday', 0)
            weekend_count = day_counts.get('Weekend and Public Holiday', 0)
            
            # Count Time of the day categories
            time_counts = site_data['Time of the day'].value_counts()
            morning_count = time_counts.get('Morning', 0)
            midday_count = time_counts.get('Midday', 0) 
            afternoon_count = time_counts.get('Afternoon', 0)
            
            site_summary.append({
                'Site name': site,
                'Weekday': weekday_count,
                'Weekend': weekend_count,
                'Morning': morning_count,
                'Midday': midday_count,
                'Afternoon': afternoon_count
            })
        
        metadata_table = pd.DataFrame(site_summary)
        metadata_table = metadata_table.sort_values('Site name')
        print(f"Loaded metadata summary for {len(metadata_table)} sites")
        return metadata_table
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return pd.DataFrame()

def load_full_metadata():
    """Load full metadata for flight log mapping"""
    try:
        if not os.path.exists(METADATA_CSV):
            return pd.DataFrame()
        
        df = pd.read_csv(METADATA_CSV)
        required_columns = ['Site name', 'Precinct', 'Day of the week', 'Hour', 'Time of the day', 'Date', 'Flight Log Filename']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in full metadata: {missing_columns}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        print(f"Error loading full metadata: {e}")
        return pd.DataFrame()

def get_flight_logs_for_site(site_name: str) -> pd.DataFrame:
    """Get flight logs for selected site"""
    full_metadata = load_full_metadata()
    if full_metadata.empty:
        return pd.DataFrame()
    
    return full_metadata[full_metadata['Site name'] == site_name][['Precinct', 'Flight Log Filename', 'Date', 'Day of the week', 'Hour', 'Time of the day']]

def load_flight_data(csv_file: str) -> pd.DataFrame:
    """Load and clean flight data from CSV"""
    file_path = os.path.join(FLIGHT_LOG_PATH, csv_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Flight log file not found: {file_path}")
    
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

def create_site_flight_visualisation(site_name: str) -> plt.Figure:
    """Create combined flight path visualisation for all flights at a site"""
    if not site_name:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Please select a site from the table', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    try:
        # Get all flight logs for this site
        flight_logs = get_flight_logs_for_site(site_name)
        
        if flight_logs.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f'No flight logs found for site: {site_name}', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 12))
        
        all_points = []
        flight_count = 0
        
        # Process each flight log
        for _, flight_row in flight_logs.iterrows():
            csv_file = flight_row['Flight Log Filename']
            
            try:
                df = load_flight_data(csv_file)
                
                if len(df) == 0:
                    continue
                
                geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=FLIGHT_CRS)
                gdf_mercator = gdf.to_crs('EPSG:3857')
                
                # Use different color for each flight
                color = plt.cm.viridis(flight_count / len(flight_logs))
                flight_name = Path(csv_file).stem.replace("-Flight-Airdata", "")
                
                # Plot flight path
                if len(gdf) > 1:
                    coords = [(row.geometry.x, row.geometry.y) for _, row in gdf.iterrows()]
                    flight_path = LineString(coords)
                    path_gdf = gpd.GeoDataFrame([{'geometry': flight_path}], crs=FLIGHT_CRS)
                    path_gdf_mercator = path_gdf.to_crs('EPSG:3857')
                    path_gdf_mercator.plot(ax=ax, color=color, linewidth=3, alpha=0.8, label=f'{flight_name}')
                
                # Plot points
                gdf_mercator.plot(ax=ax, color=color, markersize=20, alpha=0.7, edgecolor='white', linewidth=0.5)
                
                # Mark start and end points
                if len(gdf_mercator) > 1:
                    gdf_mercator.iloc[0:1].plot(ax=ax, color='green', markersize=100, marker='^', 
                                              alpha=0.8, edgecolor='white', linewidth=2)
                    gdf_mercator.iloc[-1:].plot(ax=ax, color='red', markersize=100, marker='v', 
                                               alpha=0.8, edgecolor='white', linewidth=2)
                
                all_points.extend(gdf_mercator.geometry.tolist())
                flight_count += 1
                
            except Exception as e:
                print(f"Error processing flight {csv_file}: {e}")
                continue
        
        if not all_points:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f'No valid flight data found for site: {site_name}', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        # Calculate bounds for all flights
        combined_gdf = gpd.GeoDataFrame(geometry=all_points, crs='EPSG:3857')
        bounds = combined_gdf.total_bounds
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
        
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik, alpha=1)
        
        title = f'All Flight Paths for Site: {site_name} ({flight_count} flights)'       
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Only show legend if not too many flights
        if flight_count <= 20:
            ax.legend(loc='upper right', framealpha=0.9, bbox_to_anchor=(1.05, 1))
        
        ax.set_axis_off()
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating site visualisation: {e}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

def update_flight_logs_and_plot(site_name: str):
    """Update flight logs table and plot when site is selected"""
    if not site_name:
        return pd.DataFrame(), None
    
    flight_logs = get_flight_logs_for_site(site_name)
    plot = create_site_flight_visualisation(site_name)
    return flight_logs, plot

def update_single_flight_map(flight_logs, selection: gr.SelectData):
    """Update map to show single flight when row is selected"""
    if selection is not None and not flight_logs.empty:
        csv_file = flight_logs.iloc[selection.index[0]]['Flight Log Filename']
        return create_flight_visualisation(csv_file)
    return None

def on_table_select(evt: gr.SelectData):
    """Handle site table row selection"""
    if evt.index is not None:
        metadata_df = load_metadata()
        selected_site = metadata_df.iloc[evt.index[0]]['Site name']
        plot = create_site_flight_visualisation(selected_site)
        return plot
    return None

def update_plot():
    """Update initial plot"""
    metadata_df = load_metadata()
    return metadata_df

### GRADIO INTERFACE ###

with gr.Blocks() as demo:
    gr.Markdown("# Flight Path Visualiser")
    gr.Markdown(f"Flight Log Directory: `{FLIGHT_LOG_PATH}`")
    gr.Markdown(f"Metadata file: `{METADATA_CSV}`")
    
    with gr.Row():
        # Site metadata table with value counts
        metadata_table = gr.Dataframe(
            label="Site Metadata Summary - click a row to view all flights",
            headers=["Site name", "Weekday", "Weekend", "Morning", "Midday", "Afternoon"],
            datatype=["str", "number", "number", "number", "number", "number"],
            interactive=False,
            wrap=True
        )
    with gr.Row():    
        # Flight path plot
        flight_plot = gr.Plot(
            label="Flight Paths",
            value=create_site_flight_visualisation(None)
        )

    # Event handlers
    demo.load(update_plot, outputs=[metadata_table])
    metadata_table.select(on_table_select, outputs=[flight_plot])

if __name__ == "__main__":
    demo.launch()
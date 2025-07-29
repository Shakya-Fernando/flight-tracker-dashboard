import gradio as gr
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import contextily as ctx
from pathlib import Path
from typing import List, Tuple, Optional

# Import styles and HTML content
from styles import (
    CUSTOM_CSS, 
    get_main_header, 
    get_stat_card,
    get_stat_card_pilot, 
    get_sites_overview_info,
    get_individual_flights_info,
    get_pilot_info 
)

### CONSTANTS ###
FLIGHT_LOG_PATH = "../data/raw/flight-logs/"
METADATA_CSV = "../data/raw/metadata/waterways-merged-precinct-metadata-20250625.csv"
FLIGHT_CRS = 'EPSG:4326'
MERCATOR_CRS = 'EPSG:3857'
REQUIRED_METADATA_COLUMNS = ['Site name', 'Precinct', 'Day of the week', 'Hour', 'Time of the day', 'Pilot', 'Date', 'Flight Log Filename']

### GLOBAL VARIABLES FOR CACHING ###
_metadata_cache = None
_flight_data_cache = {}

### DATA MANAGEMENT FUNCTIONS ###
def load_metadata() -> pd.DataFrame:
    """Load and validate metadata CSV"""
    if not os.path.exists(METADATA_CSV):
        print(f"Metadata file {METADATA_CSV} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(METADATA_CSV)
        
        # Validate required columns
        missing_columns = [col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in metadata: {missing_columns}")
            return pd.DataFrame()
        
        # Clean data
        df = df.dropna(subset=['Site name'])
        df = df[df['Site name'].str.strip() != ''] # Remove missing or empty
        
        print(f"Loaded metadata for {len(df)} flights across {df['Site name'].nunique()} sites")
        return df
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return pd.DataFrame()

def get_metadata(summary_only: bool = False) -> pd.DataFrame:
    """Load and cache metadata, optionally return summary"""
    global _metadata_cache
    if _metadata_cache is None:
        _metadata_cache = load_metadata()
    
    return create_metadata_summary(_metadata_cache) if summary_only else _metadata_cache

def create_metadata_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics by site"""
    if df.empty:
        return df
    
    summary_data = []
    for site in df['Site name'].unique():
        site_data = df[df['Site name'] == site]
        
        # Count categories
        day_counts = site_data['Day of the week'].value_counts() # Day 
        time_counts = site_data['Time of the day'].value_counts() # Time
        
        summary_data.append({
            'Site name': site,
            'Total Flights': len(site_data),
            'Weekday': day_counts.get('Weekday', 0),
            'Weekend': day_counts.get('Weekend and Public Holiday', 0),
            'Morning': time_counts.get('Morning', 0),
            'Midday': time_counts.get('Midday', 0),
            'Afternoon': time_counts.get('Afternoon', 0)
        })
    
    # Table summary sort by Site name 
    return pd.DataFrame(summary_data).sort_values('Site name')

def get_site_names() -> List[str]:
    """Get unique site names"""
    metadata = get_metadata()
    return metadata['Site name'].unique().tolist() if not metadata.empty else []

def get_flight_logs_for_site(site_name: str) -> pd.DataFrame:
    """Get flight logs for specific site"""
    metadata = get_metadata()
    if metadata.empty or not site_name:
        return pd.DataFrame()
    
    columns = ['Precinct', 'Flight Log Filename', 'Date', 'Day of the week', 'Hour', 'Time of the day']
    return metadata[metadata['Site name'] == site_name][columns]

def load_flight_data(csv_file: str) -> pd.DataFrame:
    """Load and cache flight data from CSV"""
    global _flight_data_cache
    
    if csv_file in _flight_data_cache: # Use cached version, if already loaded
        return _flight_data_cache[csv_file]
    
    file_path = os.path.join(FLIGHT_LOG_PATH, csv_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Flight log file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Filter video data if recording
        if 'isVideo' in df.columns:
            df = df[df['isVideo'] == 1]
        
        # Check coord columns
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("Required coordinate columns not found")
        
        # Clean lat/lng coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)] # Remove missing or zero coords
        
        # Filter data to get 5 coordinates every 1 minute
        time_interval = 60000  # 1 minute in milliseconds
        start_time = df['time(millisecond)'].min()
        # print(f"{start_time}")
        end_time = df['time(millisecond)'].max()
        
        desired_time_points = np.arange(start_time, end_time, time_interval)
        filtered_df = pd.DataFrame()
        
        for time_point in desired_time_points:
            closest_idx = (df['time(millisecond)'] - time_point).abs().argsort()[:5] # Find closest value to each tp
            closest_df = df.iloc[closest_idx] # Select correspondng row
            filtered_df = pd.concat([filtered_df, closest_df])
        
        # Remove duplicates
        filtered_df = filtered_df.drop_duplicates()
        
        # Store in cache
        _flight_data_cache[csv_file] = filtered_df
        return filtered_df
        
    except Exception as e:
        raise ValueError(f"Error loading flight data from {csv_file}: {e}")

### VISUALIZATION FUNCTIONS ###
def create_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert DataFrame to GeoDataFrame in Mercator projection"""
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=FLIGHT_CRS)
    return gdf.to_crs(MERCATOR_CRS)

def setup_plot() -> Tuple[plt.Figure, plt.Axes]:
    """Setup matplotlib figure and axes"""
    plt.close('all') # Close previous plots
    return plt.subplots(figsize=(15, 12))

def plot_flight_path(ax: plt.Axes, gdf_mercator: gpd.GeoDataFrame, 
                    flight_name: str, color=None) -> None:
    """Plot flight path, points, and markers"""
    if color is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(gdf_mercator)))
        color = 'red'  # Default path color
    else:
        colors = [color] * len(gdf_mercator)
    
    # Plot flight path - line
    if len(gdf_mercator) > 1:
        coords = [(row.geometry.x, row.geometry.y) for _, row in gdf_mercator.iterrows()]
        flight_path = LineString(coords)
        path_gdf = gpd.GeoDataFrame([{'geometry': flight_path}], crs=MERCATOR_CRS)
        path_gdf.plot(ax=ax, color=color, linewidth=3, alpha=0.8, label=flight_name)
    
    # Draw each GPS point
    point_size = 30 if color == 'red' else 20
    gdf_mercator.plot(ax=ax, color=colors, markersize=point_size, alpha=0.7, 
                     edgecolor='white', linewidth=0.5)
    
    # Plot start/end markers
    if len(gdf_mercator) > 1:
        marker_size = 150 if color == 'red' else 100
        gdf_mercator.iloc[0:1].plot(ax=ax, color='green', markersize=marker_size, 
                                   marker='^', alpha=0.8, edgecolor='white', linewidth=2,
                                   label='Start' if color == 'red' else "")
        gdf_mercator.iloc[-1:].plot(ax=ax, color='red', markersize=marker_size, 
                                   marker='v', alpha=0.8, edgecolor='white', linewidth=2,
                                   label='End' if color == 'red' else "")

def set_plot_bounds(ax: plt.Axes, gdfs: List[gpd.GeoDataFrame]) -> None:
    """Set plot bounds based on all GeoDataFrames"""
    all_bounds = [gdf.total_bounds for gdf in gdfs]
    min_x = min(bounds[0] for bounds in all_bounds)
    min_y = min(bounds[1] for bounds in all_bounds)
    max_x = max(bounds[2] for bounds in all_bounds)
    max_y = max(bounds[3] for bounds in all_bounds)
    
    centre_x, centre_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    width, height = max_x - min_x, max_y - min_y
    
    # Apply buffer and aspect ratio
    buffer = 0.1
    width_buffered = width * (1 + buffer)
    height_buffered = height * (1 + buffer)
    
    # Sspect ratio for better visuals
    fig_aspect = 4/3
    if width_buffered / height_buffered > fig_aspect:
        final_width = width_buffered
        final_height = width_buffered / fig_aspect
    else:
        final_height = height_buffered
        final_width = height_buffered * fig_aspect
    
    ax.set_xlim(centre_x - final_width/2, centre_x + final_width/2)
    ax.set_ylim(centre_y - final_height/2, centre_y + final_height/2)

def finalize_plot(ax: plt.Axes, title: str, show_legend: bool = True) -> None:
    """Add basemap, title, legend and finalize plot"""
    # ctx.add_basemap(ax, crs=MERCATOR_CRS, source=ctx.providers.OpenStreetMap.Mapnik, alpha=1)
    ctx.add_basemap(ax, crs=MERCATOR_CRS, source=ctx.providers.CartoDB.Voyager , alpha=1)
 
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.9, bbox_to_anchor=(1.05, 1))
    
    ax.set_axis_off()
    plt.tight_layout()

def create_empty_plot(message: str) -> plt.Figure:
    """Create empty plot with message"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    return fig

def create_single_flight_plot(csv_file: str) -> plt.Figure:
    """Create visualization for single flight"""
    if not csv_file:
        return create_empty_plot('No CSV file selected')
    
    try:
        df = load_flight_data(csv_file)
        if len(df) == 0:
            raise ValueError("No valid coordinate data")
        
        flight_name = Path(csv_file).stem.replace("-Flight-Airdata", "")
        gdf_mercator = create_geodataframe(df)
        
        fig, ax = setup_plot()
        plot_flight_path(ax, gdf_mercator, flight_name)
        set_plot_bounds(ax, [gdf_mercator])
        finalize_plot(ax, f'Flight Path: {flight_name}')
        
        return fig
        
    except Exception as e:
        print(f"Error creating single flight visualization: {e}")
        return create_empty_plot(f'Error: {str(e)}')

def create_site_flights_plot(site_name: str) -> plt.Figure:
    """Create combined visualization for all flights at a site"""
    if not site_name:
        return create_empty_plot('Please select a site from the table')
    
    try:
        flight_logs = get_flight_logs_for_site(site_name)
        if flight_logs.empty:
            return create_empty_plot(f'No flight logs found for site: {site_name}')
        
        fig, ax = setup_plot()
        all_gdfs = [] # Stores the data for each plotted flight
        flight_count = 0
        
        for _, flight_row in flight_logs.iterrows():
            csv_file = flight_row['Flight Log Filename']
            
            try:
                df = load_flight_data(csv_file)
                if len(df) == 0:
                    continue # Skip if no data
                
                gdf_mercator = create_geodataframe(df)
                color = plt.cm.viridis(flight_count / len(flight_logs))
                flight_name = Path(csv_file).stem.replace("-Flight-Airdata", "")
                
                plot_flight_path(ax, gdf_mercator, flight_name, color=color)
                all_gdfs.append(gdf_mercator)
                flight_count += 1
                
            except Exception as e:
                print(f"Error processing flight {csv_file}: {e}")
                continue
        
        if not all_gdfs:
            return create_empty_plot(f'No valid flight data found for site: {site_name}')
        
        set_plot_bounds(ax, all_gdfs)
        title = f'Site: {site_name} ({flight_count} flights)'
        finalize_plot(ax, title, show_legend=(flight_count <= 30)) # show legend if less than 30 flights
        
        return fig
        
    except Exception as e:
        print(f"Error creating site visualization: {e}")
        return create_empty_plot(f'Error: {str(e)}')

### EVENT HANDLERS ###
def on_site_table_select(evt: gr.SelectData) -> Optional[plt.Figure]:
    """Handle site table row selection"""
    if evt.index is not None:
        metadata_df = get_metadata(summary_only=True)
        selected_site = metadata_df.iloc[evt.index[0]]['Site name']
        return create_site_flights_plot(selected_site)
    return None

def update_flight_logs_and_plot(site_name: str) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """Update flight logs table and plot when site is selected"""
    if not site_name:
        return pd.DataFrame(), None
    
    flight_logs = get_flight_logs_for_site(site_name)
    if flight_logs.empty:
        return flight_logs, None
    
    # Show first flight by default
    csv_file = flight_logs.iloc[0]['Flight Log Filename']
    plot = create_single_flight_plot(csv_file)
    return flight_logs, plot

def update_single_flight_map(flight_logs: pd.DataFrame, selection: gr.SelectData) -> Optional[plt.Figure]:
    """Update map to show single flight when row is selected"""
    if selection is not None and not flight_logs.empty:
        csv_file = flight_logs.iloc[selection.index[0]]['Flight Log Filename']
        return create_single_flight_plot(csv_file)
    return None

# Statistics Summary
def get_summary_stats():
    metadata = get_metadata()
    if metadata.empty:
        return "0", "0", "0", "0", "0", "0", "0"
            
    # Totals
    total_flights = len(metadata)
    total_sites = metadata['Site name'].nunique()

    # Day
    weekday_flights = len(metadata[metadata['Day of the week'] == 'Weekday'])
    weekend_flights = len(metadata[metadata['Day of the week'] == 'Weekend and Public Holiday'])

    # Time
    morning_flights = len(metadata[metadata['Time of the day'] == 'Morning'])
    midday_flights = len(metadata[metadata['Time of the day'] == 'Midday'])
    afternoon_flights = len(metadata[metadata['Time of the day'] == 'Afternoon'])

    return str(total_flights), str(total_sites), str(weekday_flights), str(weekend_flights), str(morning_flights), str(midday_flights), str(afternoon_flights)

# Pilot summary
def get_pilot_summary():
    metadata = get_metadata()
    if metadata.empty:
        return [], pd.DataFrame()
        
    pilot_counts = metadata['Pilot'].value_counts().to_dict()
    pilot_stats = [{"pilot": pilot, "flights": count} for pilot, count in pilot_counts.items()]
        
    pilot_flights = metadata[['Pilot', 'Site name', 'Date', 'Flight Log Filename', 'Day of the week', 'Time of the day']]
        
    return pilot_stats, pilot_flights

### GRADIO INTERFACE ###
def create_interface():
    """Create and configure Gradio interface"""
    
    with gr.Blocks(css=CUSTOM_CSS, title="Drone Flight Tracker", theme=gr.themes.Soft()) as interface:
        gr.HTML(get_main_header())
        
        # Summar Cards
        with gr.Row():
            total_flights, total_sites, weekday_flights, weekend_flights, morning_flights, midday_flights, afternoon_flights = get_summary_stats()
            
            with gr.Column(scale=1):
                gr.HTML(get_stat_card(total_flights, "Total Flights"))
            
            with gr.Column(scale=1):
                gr.HTML(get_stat_card(total_sites, "Sites Monitored"))
            
            with gr.Column(scale=1):
                gr.HTML(get_stat_card(weekday_flights, "Weekday Flights"))
            
            with gr.Column(scale=1):
                gr.HTML(get_stat_card(weekend_flights, "Weekend Flights"))

            with gr.Column(scale=1):
                gr.HTML(get_stat_card(morning_flights, "Morning Flights"))

            with gr.Column(scale=1):
                gr.HTML(get_stat_card(midday_flights, "Midday Flights"))

            with gr.Column(scale=1):
                gr.HTML(get_stat_card(afternoon_flights, "Afternoon Flights"))

        # Tab 1: Site Overview
        with gr.Tab("Sites Overview", elem_classes=["tab-content"]):
            gr.HTML(get_sites_overview_info())
            
            with gr.Row():
                with gr.Column(scale=1):
                    metadata_table = gr.Dataframe(
                        headers=["Site", "Total", "Weekday", "Weekend", "Morning", "Midday", "Afternoon"],
                        datatype=["str", "number", "number", "number", "number", "number", "number"],
                        interactive=False,
                        wrap=True,
                        elem_classes=["dataframe"],
                        # label=" Flight Summary by Site"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    flight_plot = gr.Plot(
                        label="All Flight Paths for Selected Site",
                        value=create_site_flights_plot(None),
                        elem_classes=["plot-container"]
                    )
        
        # Tab 2: Individual Flight Analysis
        with gr.Tab("Individual Flights", elem_classes=["tab-content"]):
            gr.HTML(get_individual_flights_info())
            
            site_names = get_site_names()
            
            with gr.Row():
                with gr.Column(scale=1):
                    site_dropdown = gr.Dropdown(
                        label="Select Site", 
                        choices=site_names, 
                        value=site_names[0] if site_names else None,
                        info="Choose a site to view available flight logs"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    flight_logs_table = gr.Dataframe(
                        # label="Available Flight Logs",
                        headers=["Precinct", "Flight Log", "Date", "Day", "Hour", "Time Period"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True, 
                        elem_classes=["dataframe"],
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    individual_flight_plot = gr.Plot(
                        label="Individual Flight Path",
                        elem_classes=["plot-container"]
                    )

        # Tab 3: Pilots
        with gr.Tab("Pilots", elem_classes=["tab-content"]):
            gr.HTML(get_pilot_info())
            
            pilot_stats, pilot_flights_df = get_pilot_summary()
            with gr.Row():
                pilot_buttons = []
                for pilot_stat in pilot_stats:
                    with gr.Column(scale=1):
                        gr.HTML(get_stat_card_pilot(str(pilot_stat["flights"]), pilot_stat["pilot"]))
                        pilot_button = gr.Button("View flights", elem_classes=["gr-button-primary"])
                        pilot_buttons.append((pilot_button, pilot_stat["pilot"]))

            gr.HTML("<br>")

            with gr.Row():
                pilot_flights_table = gr.Dataframe(
                    headers=["Pilot", "Site", "Date", "Flight Log", "Day", "Time Period"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    elem_classes=["dataframe"],
                )
            
            for pilot_button, pilot_name in pilot_buttons:
                pilot_button.click(
                    lambda pilot_name=pilot_name: pilot_flights_df[pilot_flights_df['Pilot'] == pilot_name][['Pilot', 'Site name', 'Date', 'Flight Log Filename', 'Day of the week', 'Time of the day']],
                    outputs=[pilot_flights_table]
                )

        # Event handlers
        metadata_table.select(on_site_table_select, outputs=[flight_plot])
        site_dropdown.change(
            update_flight_logs_and_plot, 
            inputs=site_dropdown, 
            outputs=[flight_logs_table, individual_flight_plot]
        )
        flight_logs_table.select(
            update_single_flight_map, 
            inputs=[flight_logs_table], 
            outputs=[individual_flight_plot]
        )
        
        # Load initial data
        interface.load(lambda: get_metadata(summary_only=True), outputs=[metadata_table])
        
        if site_names:
            interface.load(
                lambda: update_flight_logs_and_plot(site_names[0]), 
                outputs=[flight_logs_table, individual_flight_plot]
            )

    return interface
    
### MAIN ###
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
else:
    demo = create_interface()
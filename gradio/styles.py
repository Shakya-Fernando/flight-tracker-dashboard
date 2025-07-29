# styles.py - HTML and CSS styling for Drone Flight Tracker
   
# CSS

CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}

.main-header {
    text-align: center;
    padding: 20px 0;
    background: #667eea;
    color: white;
    border-radius: 10px;
    margin-bottom: 30px;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 300;
}

.main-header p {
    margin: 10px 0 0 0;
    font-size: 1.1em;
    opacity: 0.9;
}

.tab-content {
    padding: 10px;
    background: #f8f9fa;
}

.info-card {
    background: white;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin-bottom: 10px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.stat-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
    border-top: 3px solid #667eea;
}

.stat-number {
    font-size: 2em;
    font-weight: bold;
    color: #667eea;
    margin: 0;
}

.stat-label {
    color: #666;
    margin: 5px 0 0 0;
    font-size: 1em;
}

.stat-card-pilot {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
    border-top: 3px solid #9d66ea;
}

.stat-number-pilot {
    font-size: 2em;
    font-weight: bold;
    color: #9d66ea;
    margin: 0;
}

.dataframe {
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.plot-container {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 10px 0;
}

.gr-button-primary {
    background: #dbc2ff !important;
    border: none !important;
    border-radius: 6px !important;
    width: 150px !important;
    margin: auto !important;
    display: block !important;
}

.gr-dropdown {
    border-radius: 6px !important;
}
"""

# HTML
def get_main_header():
    return """
    <div class="main-header">
        <h1>Drone Flight Tracker</h1>
    </div>
    """

def get_stat_card(number, label):
    return f"""
    <div class="stat-card">
        <p class="stat-number">{number}</p>
        <p class="stat-label">{label}</p>
    </div>
    """

def get_stat_card_pilot(number, label):
    return f"""
    <div class="stat-card-pilot">
        <p class="stat-number-pilot">{number}</p>
        <p class="stat-label">{label}</p>
    </div>
    """    

def get_sites_overview_info():
    return """
    <div class="info-card">
        <h3>Sites Overview</h3>
        <p>Click on any row to visualise all flight paths for that Site.</p>
    </div>
    """

def get_individual_flights_info():
    return """
    <div class="info-card">
        <h3>Individual Flight Analysis</h3>
        <p>Select a Site to view all available flights, then click on any flight to see its specific route.</p>
    </div>
    """

def get_pilot_info():
    return """
    <div class="info-card">
        <h3> Pilot Flight Statistics </h3>
        <p> View the number of drone flights completed by each pilot </p>
    </div>
    """    
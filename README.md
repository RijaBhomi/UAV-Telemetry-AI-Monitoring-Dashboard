# UAV Telemetry & AI Monitoring Dashboard
 
Analytical dashboard for the **Fixed-Wing UAV Reinforcement Adaptive Co-Piloting** capstone project.

**LIVE DEMO:** https://uav-telemetry-ai-monitoring-dashboard-dwxdwcd7phn3pkbytjan6d.streamlit.app/

## What this dashboard shows
 
| Section | What it does |
|---|---|
| Live flight telemetry | Altitude, airspeed, pitch, roll, heading, throttle — metric cards + line graphs |
| AI co-pilot monitoring | Reward convergence, reward components (R_task / R_smooth / R_safety), envelope violations |
| Flight log table | Filterable table of all telemetry rows with status indicators |
| Session summary | Max/avg stats + CSV download |
 
---

## Running locally
 
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
 
---
## Using real flight data
 
Once JSBSim + MAVLink is running, export telemetry as a CSV with these columns:
 
```
timestamp, altitude_m, airspeed_kmh, pitch_deg, roll_deg,
heading_deg, throttle_pct, flight_mode, status
```
 
Then use the **Load CSV file** option in the sidebar to upload it.
 
---
 
## Reward weight tuning
 
Use the sidebar sliders to adjust **w1 / w2 / w3** (must sum to 1.0).  
The total reward chart updates instantly so we can see how weight changes affect convergence.
 
---

# =============================================================
# F1 LAP TIME PREDICTOR - PROJECT SUMMARY
# =============================================================
# WHAT THIS DOES:
# Pulls George Russell's race data from 2022-2025 using the
# fastf1 API, builds a feature dataset, and trains an XGBoost
# model to predict his lap times at each circuit.
#
# WHAT WE LEARNED:
# - With sector times included: MAE of ~0.7s (very accurate,
#   but essentially cheating since lap = s1 + s2 + s3)
# - Without sector times: MAE jumps to ~6s (much less useful)
# - Circuit encoding as a plain number gives the model no real
#   information about WHY lap times differ between tracks
#
# LIMITATIONS / WHY THIS DOESN'T REFLECT REAL F1 MODELLING:
# - Real teams use physics/CFD simulation, not historical ML
# - Real predictions are made 24-48hrs out using practice data
#   not a full year in advance
# - 2026 reg changes make historical data even less relevant
# - Would need track characteristics (length, corners, speed)
#   as features to be genuinely useful
#
# POTENTIAL IMPROVEMENTS:
# - Add circuit metadata as features (track length etc.)
# - Predict from Friday practice data for realistic use case
# - Train per-circuit models rather than one global model
# =============================================================

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import fastf1 as ff1
# STEP 1 - imports done!

# STEP 2 - parameters
year = 2025
wknd = 'Austrian Grand Prix'
sesh = 'R'
driver = 'RUS'
colormap = mpl.cm.plasma

# STEP 3 - load session data
session = ff1.get_session(year, wknd, sesh)
session.load()
weekend = session.event
print(f"Loaded {weekend['EventName']} {session.name} session")

# STEP 4 - pick fastest lap for driver
fastest_lap = session.laps.pick_drivers(driver).pick_fastest()

# STEP 5 - get telemetry and create line segments
telemetry = fastest_lap.get_telemetry()

points = np.array([telemetry['X'], telemetry['Y']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# STEP 6 - create plot
fig, ax = plt.subplots(figsize=(12, 6.75))
fig.suptitle(f"{weekend['EventName']} {year} - {driver} - Speed", size=24, y=0.97)
ax.plot(telemetry['X'], telemetry['Y'], color='black', linewidth=16, zorder=0)
norm = plt.Normalize(telemetry['Speed'].min(), telemetry['Speed'].max())
lc = LineCollection(segments, cmap=colormap, norm=norm, linewidth=8, zorder=1)
lc.set_array(telemetry['Speed'])
ax.add_collection(lc)

# add colorbar so speed scale is visible
cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.02])
plt.colorbar(lc, cax=cbaxes, orientation='horizontal', label='Speed (km/h)')
#misc graph settings
ax.axis('off')
plt.subplots_adjust(left=-0.1, right=0.9, top=0.9, bottom=0.12)
plt.show()

#########################################

#SETUP: gathering and cleaning raw data for ml model training
##^collect the fastest lap data for RUS across multiple seasons and save it to a CSV file
import fastf1 as ff1
import pandas as pd
import os

os.makedirs('cache', exist_ok=True)
ff1.Cache.enable_cache('cache')

driver = 'RUS'
seasons = [2022, 2023, 2024, 2025]
all_laps = []

for year in seasons:
    schedule = ff1.get_event_schedule(year, include_testing=False)
    
    for _, event in schedule.iterrows():
        if event['EventDate'] > pd.Timestamp.now():
            continue
        
        try:
            session = ff1.get_session(year, event['EventName'], 'R')
            session.load(telemetry=False, weather=True, messages=False)
            
            driver_laps = session.laps.pick_drivers(driver)
            if driver_laps.empty:
                continue
            
            fastest = driver_laps.pick_fastest()
            weather = session.weather_data.mean(numeric_only=True)
            
            all_laps.append({
                'year':       year,
                'circuit':    event['EventName'],
                'lap_time_s': fastest['LapTime'].total_seconds(),
                'sector1_s':  fastest['Sector1Time'].total_seconds(),
                'sector2_s':  fastest['Sector2Time'].total_seconds(),
                'sector3_s':  fastest['Sector3Time'].total_seconds(),
                'compound':   fastest['Compound'],
                'tyre_age':   fastest['TyreLife'],
                'air_temp':   weather['AirTemp'],
                'track_temp': weather['TrackTemp'],
                'humidity':   weather['Humidity'],
            })
            
            print(f"✓ {year} {event['EventName']}")
        
        except Exception as e:
            print(f"✗ {year} {event['EventName']} — skipped ({e})")

df = pd.DataFrame(all_laps)
df.to_csv('rus_data.csv', index=False)
print(f"\nDone! {len(df)} races collected.")
print(df.head()) 

#checkin the CSV
import pandas as pd

df = pd.read_csv('rus_data.csv')
print(df.shape)        # how many rows and columns
print(df.head(10))     
print(df.dtypes)       # column types
print(df.isnull().sum()) # any missing values?

#fill null values in compound with 'UNKNOWN' and encode it as numbers for model
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('rus_data.csv')

# fix missing compound
df['compound'] = df['compound'].fillna('UNKNOWN')

# encode compound as numbers (SOFT=2, MEDIUM=1, HARD=0 roughly)
compound_encoder = LabelEncoder()
df['compound_enc'] = compound_encoder.fit_transform(df['compound'])

# encode circuit as numbers
circuit_encoder = LabelEncoder()
df['circuit_enc'] = circuit_encoder.fit_transform(df['circuit'])

# drop original text columns
df = df.drop(columns=['circuit', 'compound'])

print(df.head())
print(df.dtypes)
print(df.isnull().sum())


#ML TIME!!
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# train/test split by season
train = df[df['year'] <= 2024]
test = df[df['year'] == 2025]

features = ['year', 'tyre_age', 'air_temp', 'track_temp', #omitted sector times as highly correlated w lap time/ would make the model 2ez2 predict
            'humidity', 'compound_enc', 'circuit_enc']
target = 'lap_time_s'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# fit model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae:.3f} seconds")

# see actual vs predicted
results = test[['year', 'circuit_enc']].copy()
results['actual'] = y_test.values
results['predicted'] = preds
results['diff'] = (results['predicted'] - results['actual']).round(3)
print(results)

#findingd: What this means is that sector times were doing almost all the heavy lifting in the first model. 
#Without them, circuit + weather + tyre alone aren't enough to predict lap time accurately.
#this makes intuitive sense because lap time varies so much between circuits and 
#our remaining features don't fully capture what makes each circuit unique beyond just the circuit ID number.
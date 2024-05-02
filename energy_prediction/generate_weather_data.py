import pandas as pd
import numpy as np

# Define the time range
dates = pd.date_range(start="2023-01-01", periods=72, freq='H')  # 3 days hourly

# Generate synthetic weather data
np.random.seed(42)  # For reproducibility
ghi = np.random.uniform(0, 1000, size=(72,))  # GHI might peak at about 1000 W/m² during midday
dni = ghi * np.random.uniform(0.7, 1.0, size=(72,))  # DNI is generally equal or less than GHI
dhi = ghi - dni  # DHI is the difference in this simple model
temp_air = np.random.uniform(-5, 15, size=(72,))  # Temperatures between -5 and 15 degrees Celsius
wind_speed = np.random.uniform(0, 15, size=(72,))  # Wind speeds in m/s

# Create DataFrame
weather_data = pd.DataFrame({
    'timestamp': dates,
    'ghi': ghi,
    'dni': dni,
    'dhi': dhi,
    'temp_air': temp_air,
    'wind_speed': wind_speed
})

# Save to CSV
weather_data.to_csv('energy_prediction\energy_data\historical_weather.csv', index=False)

print(weather_data.head())
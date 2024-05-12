import pandas as pd
import numpy as np
from pvlib import location, irradiance, temperature, pvsystem
from random import choice, randint, seed
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from sandiapv_energy_prediction import * 

# Function to load weather data
def load_weather_data(file_path):
    return pd.read_csv(file_path)

# Function to generate a specified number of random panel configurations
def generate_random_panels(num_panels):
    panel_types = ['Canadian_Solar_CS5P_220M___2009_']
    tilts = [randint(5, 40) for _ in range(num_panels)]
    azimuths = [randint(90, 270) for _ in range(num_panels)]
    module_types = ['monocrystalline', 'polycrystalline', 'thin-film', 'bifacial']
    return [{'type': choice(panel_types), 'tilt': tilt, 'azimuth': azimuth, 'module_type': choice(module_types)}
            for tilt, azimuth in zip(tilts, azimuths)]

# Function to simulate PV output for random days
def simulate_pv_output_random_days(system, weather_data, location, num_days=10):
    days = np.random.choice(range(365), num_days, replace=False) * 24  # Pick random days and convert to indices
    results = {}
    for day in days:
        start_idx = day
        end_idx = day + 24
        daily_weather = weather_data.iloc[start_idx:end_idx]
        solar_position = location.get_solarposition(daily_weather.index)
        temp_air = daily_weather['temp_air']
        wind_speed = daily_weather['wind_speed']
        output_data = pd.DataFrame(index=daily_weather.index)
        for array in system.arrays:
            poa_irrad = irradiance.get_total_irradiance(
                array.mount.surface_tilt, array.mount.surface_azimuth,
                solar_position['apparent_zenith'], solar_position['azimuth'],
                daily_weather['dni'], daily_weather['Q'], daily_weather['dhi']
            )
            cell_temperature = temperature.sapm_cell(
                poa_irrad['poa_global'], temp_air, wind_speed, **array.temperature_model_parameters
            )
            dc_output = system.pvwatts_dc(poa_irrad['poa_global'], cell_temperature)
            output_data[array.name] = dc_output
        results[day // 24] = output_data.sum(axis=0).to_dict()
    return results

# Main block to generate data and save to CSV
if __name__ == "__main__":
    seed(0)
    weather_data = load_weather_data('energy_prediction/energy_data/historical_weather.csv')
    weather_data.index = pd.to_datetime(weather_data['date_time'])
    site_location = location.Location(latitude=52.52, longitude=13.4050, altitude=34, tz='Europe/Amsterdam')
    panels = generate_random_panels(100)  # Generate 100 random panel configurations

    all_panel_results = []
    for panel in panels:
        system = get_pv_system(panel)
        panel_results = simulate_pv_output_random_days(system, weather_data, site_location)
        for day, outputs in panel_results.items():
            row = {
                'panel_type': panel['type'],
                'tilt': panel['tilt'],
                'azimuth': panel['azimuth'],
                'module_type': panel['module_type'],
                'day': day,
                **outputs
            }
            all_panel_results.append(row)
    
    df = pd.DataFrame(all_panel_results)
    df.to_csv('energy_prediction/energy_data/energy_output_random_days.csv', index=False)

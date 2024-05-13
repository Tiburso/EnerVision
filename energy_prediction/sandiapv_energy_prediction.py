import pandas as pd
import numpy as np
from pvlib import location, irradiance, temperature, pvsystem
from random import choice, randint, seed
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapz


def load_weather_data(file_path):
    """ Load historical weather data from a CSV file. """
    return pd.read_csv(file_path)

def generate_random_panels(num_panels):
    """ Generate random configurations for a number of PV panels. """
    #panel_types = ['Canadian_Solar_CS5P_220M___2009_']
    tilts = [randint(5, 40) for _ in range(num_panels)]  # Tilt angles between 5 and 40 degrees
    azimuths = [randint(90, 270) for _ in range(num_panels)]  # Southward orientations between East and West
    module_types = ['monocrystalline', 'polycrystalline', 'thin-film', 'bifacial']
    return [{'type': choice(panel_types), 'tilt': tilt, 'azimuth': azimuth, 'module_type': choice(module_types)} 
            for tilt, azimuth in zip(tilts, azimuths)]

def get_pv_system(panel):
    """ Retrieves and configures a PVSystem object based on the panel type and parameters. """
    #module = pvsystem.retrieve_sam('SandiaMod')[panel['type']]
   
    module_specs = {
    'monocrystalline': {'pdc0': 220, 'gamma_pdc': -0.0045},
    'polycrystalline': {'pdc0': 200, 'gamma_pdc': -0.005},
    'thin-film': {'pdc0': 180, 'gamma_pdc': -0.002},
    'bifacial': {'pdc0': 210, 'gamma_pdc': -0.004}
    } 
    'lllll'
    module_parameters = module_specs[panel['module_type']]#{'pdc0': 200, 'gamma_pdc': -0.004}  
    
    mount = pvsystem.FixedMount(surface_tilt=panel['tilt'], surface_azimuth=panel['azimuth'])

    inverter_parameters = {'pdc0': 5000, 'eta_inv_nom': 0.96}
    array_one = (pvsystem.Array( mount=mount, 
                                module_parameters=module_parameters,
                                temperature_model_parameters= temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'],
                                #module_type = 'monocrystalline'
                                ))
    return pvsystem.PVSystem(name = 'system1',
                            arrays =array_one,
                            inverter =inverter_parameters,                           
                            )
    


def simulate_pv_output(system, weather_data, location):
    """ Simulate daily PV output for all arrays in the system. """
    solar_position = location.get_solarposition(weather_data.index)
    temp_air = weather_data['temp_air']
    wind_speed = weather_data['wind_speed']
    
    # Initialize an empty DataFrame to store the output for each array
    output_data = pd.DataFrame(index=weather_data.index)

    # Simulate output for each array in the system
    for array in system.arrays:
        #print(array)
        #print('____________________')
        # Calculate POA irradiance
        aoi = irradiance.aoi(array.mount.surface_tilt, array.mount.surface_azimuth, solar_position['apparent_zenith'], solar_position['azimuth'])
        poa_irrad = irradiance.get_total_irradiance(array.mount.surface_tilt, array.mount.surface_azimuth,
                                                    solar_position['apparent_zenith'], solar_position['azimuth'],
                                                    weather_data['dni'], weather_data['temp_air'], weather_data['dhi'])
        # Calculate cell temperature
        cell_temperature = temperature.sapm_cell(poa_irrad['poa_global'], temp_air, wind_speed,
                                                  **array.temperature_model_parameters)
        single_array_system = pvsystem.PVSystem(
                             name=system.name,
                            arrays=[system.arrays[0]],
                            inverter=system.inverter,)
        # Calculate DC output for the current array
        dc_output = single_array_system.pvwatts_dc(poa_irrad['poa_global'], cell_temperature)
        
        #should be: but code above is workaround due to unknown error
        #dc_output = system.pvwatts_dc(poa_irrad['poa_global'], cell_temperature)
        
        # Store the DC output in the output_data DataFrame
        output_data[array.name] = dc_output
    
    return output_data

def fit_gaussian_to_daily_data(daily_data):
    x_numeric = np.arange(len(daily_data))
    popt, _ = curve_fit(gaussian, x_numeric, daily_data, p0=[max(daily_data), np.argmax(daily_data), 1])
    return popt 

def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

def plot_energy_outputs(data, energy_outputs, days_to_plot=3):
    plt.figure(figsize=(12, 8))
    
    if not isinstance(energy_outputs.index, pd.DatetimeIndex):
        energy_outputs.index = pd.to_datetime(energy_outputs.index)

    colors = ['red', 'blue', 'green'] 
    colors2 = ['orange', 'purple', 'yellow']

    num_panels = len(energy_outputs.columns) // 4
    areas = []
    for i in range(num_panels):
        panel_energy_output = energy_outputs.iloc[:, i*4:(i+1)*4]

        for day in range(days_to_plot):
            start_idx = day * 24
            end_idx = (day + 1) * 24
            daily_data = panel_energy_output.iloc[start_idx:end_idx].mean(axis=1)

            popt = fit_gaussian_to_daily_data(daily_data)
            if popt is not None:
                x_dense = np.linspace(0, 23, 500)  
                gaussian_curve = gaussian(x_dense, *popt)

                # Calculate the area under the Gaussian curve
                area_gaussian = trapz(gaussian_curve, dx=x_dense[1]-x_dense[0])

                # Calculate the area under the original daily mean data
                area_original = trapz(daily_data, dx=1) 

                # Calculate the difference in areas
                area_difference =  area_gaussian/ area_original
                areas.append(area_difference)
                x_dense = np.linspace(0, 23, 500) 
                x_plot = pd.date_range(start=energy_outputs.index[start_idx], periods=24, freq='H')
                x_plot_dense = pd.date_range(start=x_plot[0], end=x_plot[-1], periods=500)  
                plt.plot(x_plot_dense, gaussian(x_dense, *popt), color=colors2[i], label=f'Gaussian Panel {i+1}, {area_difference:.2f} Wh', linewidth=2)


            # Plot the original daily mean energy data
            plt.plot(energy_outputs.index[start_idx:end_idx], daily_data,
                     label=f'Panel {i+1} Mean', linestyle='--', color=colors[i])
            
            # Plot W.value from data DataFrame
            # Adjust the slicing based on your needs
            start_time = energy_outputs.index[0]
            end_time = energy_outputs.index[min(len(energy_outputs.index), days_to_plot*24) - 1]
            plt.plot(data[start_time:end_time].index, data[start_time:end_time]['W.mean_value'], 
                color='black', label='W.mean_value', linewidth=2, linestyle='-.')
    print(np.mean(np.array(areas)))
    plt.title('Predicted Energy Production Over Time')
    plt.xlabel('Time')
    plt.ylabel('Energy Output (Wh)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def prepare_data_for_model(energy_outputs, weather_data, panels):
    model_data = []

    # Iterate over each panel configuration
    for i, panel in enumerate(panels):
        panel_energy_output = energy_outputs.iloc[:, i*4:(i+1)*4]
        
        # Process data day by day
        for day in range(365):  # Assuming you have a full year of data
            start_idx = day * 24
            end_idx = (day + 1) * 24

            # Extract daily weather data and energy outputs
            daily_weather = weather_data.iloc[start_idx:end_idx]
            daily_data = panel_energy_output.iloc[start_idx:end_idx].mean(axis=1)

            # Fit Gaussian to daily energy data
            popt = fit_gaussian_to_daily_data(daily_data)

            if popt is not None:
                row = {
                    'panel_type': panel['type'],
                    'tilt': panel['tilt'],
                    'azimuth': panel['azimuth'],
                    'module_type': panel['module_type'],
                    # Store sequences as lists in the DataFrame cell
                    'temperature_sequence': daily_weather['temp_air'].tolist(),
                    'wind_speed_sequence': daily_weather['wind_speed'].tolist(),
                    'dni_sequence': daily_weather['dni'].tolist(),
                    'dhi_sequence': daily_weather['dhi'].tolist(),
                    'global_irradiance_sequence': daily_weather['temp_air'].tolist(),
                    'gaussian': popt.tolist()
                }
                model_data.append(row)

    return pd.DataFrame(model_data)


if __name__ == "__main__":
    # Load weather data
    #weather_data = load_weather_data('energy_prediction/energy_data/historical_weather.csv')
    #weather_data = load_weather_data('energy_prediction/energy_data/output.csv')
    weather_data = load_weather_data('energy_prediction/energy_data/result.csv')
    #weather_data.index = pd.to_datetime(weather_data['date_time'])
    #weather_data.index = pd.to_datetime(weather_data['date_time'], utc=True)
    weather_data.index = pd.to_datetime(weather_data['time'], utc=True)
    # Define location (example: Berlin, Germany)
    site_location = location.Location(latitude=52.52, longitude=13.4050, altitude=34, tz='Europe/Amsterdam')

    # Generate random panels
    seed(0)
    panels = generate_random_panels(1)

    energy_outputs = pd.DataFrame(index=weather_data.index)

    # Simulate PV output for all panels
    results = []
    for i, panel in enumerate(panels):
        system = get_pv_system(panel)
        daily_output = simulate_pv_output(system, weather_data, site_location)
        results.append({
            'panel_type': panel['type'],
            'tilt': panel['tilt'],
            'azimuth': panel['azimuth'],
            'daily_output': daily_output.resample('D').sum()  
             })
        for column in daily_output.columns:
            energy_outputs[f'{column}_Panel_{i+1}'] = daily_output[column]
    energy_outputs.to_csv('energy_prediction/energy_data/energy_output.csv', sep=',', index=True)
    
    # Plot energy outputs
    plot_energy_outputs(weather_data, energy_outputs)
    # Assume weather_data and panels are already loaded and processed
    prepared_data = prepare_data_for_model(energy_outputs, weather_data, panels)
    prepared_data.to_csv('energy_prediction/energy_data/model_input.csv', index=False)
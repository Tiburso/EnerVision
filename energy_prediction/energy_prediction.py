import pandas as pd
import numpy as np
import pvlib
from pvlib import location, irradiance, temperature, pvsystem
from random import choice, randint, seed
import matplotlib.pyplot as plt


def load_weather_data(file_path):
    """ Load historical weather data from a CSV file. """
    return pd.read_csv(file_path)

def generate_random_panels(num_panels):
    """ Generate random configurations for a number of PV panels. """
    panel_types = ['Canadian_Solar_CS5P_220M___2009_']
    tilts = [randint(5, 40) for _ in range(num_panels)]  # Tilt angles between 5 and 40 degrees
    azimuths = [randint(90, 270) for _ in range(num_panels)]  # Southward orientations between East and West
    module_types = ['monocrystalline', 'polycrystalline', 'thin-film', 'bifacial']
    return [{'type': choice(panel_types), 'tilt': tilt, 'azimuth': azimuth, 'module_type': choice(module_types)} 
            for tilt, azimuth in zip(tilts, azimuths)]

def get_pv_system(panel):
    """ Retrieves and configures a PVSystem object based on the panel type and parameters. """
    #module = pvsystem.retrieve_sam('SandiaMod')[panel['type']]

    #Check modules make importable.
    sandia_modules = pvsystem.retrieve_sam('SandiaMod')
    module = sandia_modules[panel['type']]
    module_parameters = {'pdc0': 200, 'gamma_pdc': -0.004}  
    mount = pvsystem.FixedMount(surface_tilt=panel['tilt'], surface_azimuth=panel['azimuth'])
    inverter_parameters = {'pdc0': 5000, 'eta_inv_nom': 0.96}
    arrays = []
    
    module_types = ['monocrystalline', 'polycrystalline', 'thin-film', 'bifacial']
    for i, module_type in enumerate(module_types):
        arrays.append(pvsystem.Array(name = module_type,
                                mount=mount, 
                                module_parameters=module_parameters,
                                temperature_model_parameters= temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'],
                                module_type = module_type
                                ))
    
    return pvsystem.PVSystem(name = 'system1',
                            arrays =arrays,
                            inverter =inverter_parameters,                           
                            )
    """
    array_one = (pvsystem.Array( mount=mount, 
                                module_parameters=module_parameters,
                                temperature_model_parameters= temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'],
                                module_type = 'monocrystalline'
                                ))
    return pvsystem.PVSystem(name = 'system1',
                            arrays =array_one,
                            inverter =inverter_parameters,                           
                            )
    """


def simulate_pv_output(system, weather_data, location):
    """ Simulate daily PV output for all arrays in the system. """
    solar_position = location.get_solarposition(weather_data.index)
    temp_air = weather_data['temp_air']
    wind_speed = weather_data['wind_speed']
    
    # Initialize an empty DataFrame to store the output for each array
    output_data = pd.DataFrame(index=weather_data.index)

    # Simulate output for each array in the system

    for array in system.arrays:
        print(array)
        print('____________________')
        # Calculate POA irradiance
        aoi = irradiance.aoi(array.mount.surface_tilt, array.mount.surface_azimuth, solar_position['apparent_zenith'], solar_position['azimuth'])
        poa_irrad = irradiance.get_total_irradiance(array.mount.surface_tilt, array.mount.surface_azimuth,
                                                    solar_position['apparent_zenith'], solar_position['azimuth'],
                                                    weather_data['dni'], weather_data['ghi'], weather_data['dhi'])
        # Calculate cell temperature
        cell_temperature = temperature.sapm_cell(poa_irrad['poa_global'], temp_air, wind_speed, **array.temperature_model_parameters)
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

def plot_energy_outputs(energy_outputs):
    """ Plot the energy outputs over time for all panels as filled areas between module types. """
    plt.figure(figsize=(12, 8))
    
    # Get the number of panels
    num_panels = len(energy_outputs.columns) // 4
    
    # Iterate over each panel
    for i in range(num_panels):
        # Get the energy output for the current panel
        panel_energy_output = energy_outputs.iloc[:, i*4:(i+1)*4]
        
        # Compute the minimum and maximum values for each time step across module types
        min_values = panel_energy_output.min(axis=1)
        max_values = panel_energy_output.max(axis=1)
        mean_values = panel_energy_output.mean(axis=1)
        
        # Plot the mean energy output
        plt.plot(energy_outputs.index, mean_values, label=f'Panel {i+1}')
        # Plot the filled area between the minimum and maximum values
        plt.fill_between(energy_outputs.index, min_values, max_values, alpha=0.3, label=f'Panel {i+1}')
    
    plt.title('Predicted Energy Production Over Time')
    plt.xlabel('Time')
    plt.ylabel('Energy Output (Wh)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load weather data
    weather_data = load_weather_data('energy_prediction/energy_data/historical_weather.csv')
    weather_data.index = pd.to_datetime(weather_data['timestamp'])
    
    # Define location (example: Berlin, Germany)
    site_location = location.Location(latitude=52.52, longitude=13.4050, altitude=34, tz='Europe/Berlin')

    # Generate random panels
    seed(0)
    panels = generate_random_panels(3)

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
            'daily_output': daily_output.resample('D').sum()  # Sum daily output
        })
        for column in daily_output.columns:
            energy_outputs[f'{column}_Panel_{i+1}'] = daily_output[column]

    # Plot energy outputs
    plot_energy_outputs(energy_outputs)
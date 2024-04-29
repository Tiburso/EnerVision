import pandas as pd 
from sandiapv_energy_prediction import * 
import matplotlib.pyplot as plt


def plot_comparison(energy_outputs, measured_outputs):
    """ Plot both predicted and actual energy outputs. """
    plt.figure(figsize=(12, 8))
    
    # Plot simulated output
    for column in energy_outputs.columns:
        plt.plot(energy_outputs.index, energy_outputs[column], label=f'Simulated {column}', linestyle='-')

    # Plot measured output
    for column in measured_outputs.columns:
        plt.plot(measured_outputs.index, measured_outputs[column], label=f'Measured {column}', linestyle='--')

    plt.title('Comparison of Predicted and Actual Energy Production')
    plt.xlabel('Time')
    plt.ylabel('Energy Output (Wh)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_measurement_data(data_path,days):
    # Load the data from the CSV file
    data = pd.read_csv(data_path)

    # Convert 'Time' column to datetime and set it as the index
    data['Time'] = pd.to_datetime(data['Time'])
    data.set_index('Time', inplace=True)

    # Filter the data to only include the first three days
    start_date = data.index.min()
    end_date = start_date + pd.DateOffset(days =days)
    data_first_three_days = data[start_date:end_date]

    # Plot the energy production
    plt.figure(figsize=(12, 6))
    plt.plot(data_first_three_days.index, data_first_three_days['PV Productie (W)'], label='PV Production', color='blue')
    plt.title('PV Energy Production Over the First Three Days')
    plt.xlabel('Time')
    plt.ylabel('Production (W)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Existing code for loading and simulating data...
    
    # plot measured data
    plot_measurement_data('energy_prediction/energy_data/Systeem_Amstelveen/2022_15min_data.csv',days =4 )

    # Plotting both outputs
    #plot_comparison(energy_outputs, measured_data)
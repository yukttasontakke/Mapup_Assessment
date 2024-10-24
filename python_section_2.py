
import pandas as pd
import numpy as np
from datetime import time

#Question 9
def calculate_distance_matrix(file_path):
    df = pd.read_csv(file_path)

    # Initialize the distance matrix with infinity
    locations = df['id'].unique()
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)

    # Fill the diagonal with zeros
    np.fill_diagonal(distance_matrix.values, 0)

    # Fill the matrix with the known distances
    for _, row in df.iterrows():
        distance_matrix.at[row['id'], row['id_2']] = row['distance']
        distance_matrix.at[row['id_2'], row['id']] = row['distance']  # Bidirectional distance

    # Compute the cumulative distances
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    
    return distance_matrix

file_path = 'dataset-2.csv'
distance_matrix = calculate_distance_matrix(file_path)
print(distance_matrix)

#Question 10
def unroll_distance_matrix(distance_matrix):
    unrolled_data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                unrolled_data.append((id_start, id_end, distance_matrix.at[id_start, id_end]))

    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df
    
unrolled_df = unroll_distance_matrix(distance_matrix)
print

#Question 11
def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Calculate the average distance for the reference id
    reference_distances = df[df['id_start'] == reference_id]['distance']
    avg_distance = reference_distances.mean()
    
    # Define the 10% threshold range
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1
    
    # Find ids within the threshold range
    within_threshold = df.groupby('id_start')['distance'].mean().between(lower_bound, upper_bound)
    result = within_threshold[within_threshold].index.tolist()
    
    # Remove the reference id from the list and sort the remaining ids
    if reference_id in result:
        result.remove(reference_id)
    result.sort()
    
    return result

reference_id = 1  # replace with actual reference id from the DataFrame
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result_ids)

#Question 12
def calculate_toll_rate(df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    
    return df

unrolled_df_with_rates = calculate_toll_rate(unrolled_df)
print(unrolled_df_with_rates)

#Question 13
def calculate_time_based_toll_rates(df):
    # Define the discount factors
    weekday_factors = [
        (time(0, 0), time(10, 0), 0.8),
        (time(10, 0), time(18, 0), 1.2),
        (time(18, 0), time(23, 59, 59), 0.8)
    ]
    weekend_factor = 0.7

    # Initialize lists to hold new columns
    start_days = []
    start_times = []
    end_days = []
    end_times = []

    vehicle_types = ['moto', 'car', 'rv', 'bus', 'truck']

    # Copy original vehicle rates
    for vehicle in vehicle_types:
        df[f'{vehicle}_original'] = df[vehicle]
    
    def apply_weekday_discounts(start_day, start_time, vehicle):
        for (start, end, factor) in weekday_factors:
            if start <= start_time < end:
                return df[vehicle] * factor
        return df[vehicle]

    # Iterate over each day of the week and time ranges
    for id_start, id_end, distance in df.itertuples(index=False):
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            for start, end in weekday_factors:
                start_days.append(day)
                start_times.append(start)
                end_days.append(day)
                end_times.append(end)
                for vehicle in vehicle_types:
                    if day in ['Saturday', 'Sunday']:
                        df[vehicle] *= weekend_factor
                    else:
                        df[vehicle] = apply_weekday_discounts(day, start, vehicle)
    
    # Add new columns to DataFrame
    df['start_day'] = start_days
    df['start_time'] = start_times
    df['end_day'] = end_days
    df['end_time'] = end_times

    return df

df_with_time_based_rates = calculate_time_based_toll_rates(unrolled_df_with_rates)
print(df_with_time_based_rates)

from typing import Dict, List
from itertools import permutations
from math import radians, sin, cos, sqrt, atan2
import polyline
import pandas as pd

#Question 1
def reverse_by_n(lst, n):
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        # Create a temporary list to hold the current group
        temp = []
        for j in range(i, min(i + n, length)):
            temp.append(lst[j])
        
        # Reverse the temporary list manually
        for k in range(len(temp) - 1, -1, -1):
            result.append(temp[k])
    
    return result
print(reverse_by_n([1, 2, 3, 4, 5, 6, 7, 8], 3)) 
print(reverse_by_n([1, 2, 3, 4, 5], 2))
print(reverse_by_n([10, 20, 30, 40, 50, 60, 70], 4))

#Question 2
def group_by_length(strings):
    length_dict = {}
    
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Sort the dictionary by keys (lengths)
    sorted_length_dict = dict(sorted(length_dict.items()))
    
    return sorted_length_dict
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))

#Question 3
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict({f"{k}[{i}]": item}, parent_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)

#Question 4
def unique_permutations(lst):
    # Generate all permutations
    all_perms = permutations(lst)
    # Use a set to filter out duplicate permutations
    unique_perms = set(all_perms)
    # Convert each tuple back to a list
    return [list(perm) for perm in unique_perms]

input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)

#Question 5
def find_all_dates(text):
    # Regular expression pattern to match the date formats
    date_pattern = r'\b(?:\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # Find all matches in the text
    dates = re.findall(date_pattern, text)
    
    return dates

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)

#Question 6
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def decode_polyline_to_df(polyline_str):
    # Decode the polyline string
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate distances
    distances = [0]  # First row has a distance of 0
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    # Add the distances to the DataFrame
    df['distance'] = distances
    
    return df

polyline_str = 'u{~vFvyys@fS]'
df = decode_polyline_to_df(polyline_str)
print(df)

#Question 7
def rotate_matrix_90_clockwise(matrix):
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    return rotated_matrix

def transform_matrix(matrix):
    n = len(matrix)
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(matrix[i]) - matrix[i][j]
            col_sum = sum(matrix[k][j] for k in range(n)) - matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum
    
    return transformed_matrix

def rotate_and_transform(matrix):
    rotated_matrix = rotate_matrix_90_clockwise(matrix)
    final_matrix = transform_matrix(rotated_matrix)
    return final_matrix

#Question 8
def verify_time_completeness(df):
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    unique_pairs = df.groupby(['id', 'id_2'])
    
    result = []

    for (id_, id_2), group in unique_pairs:
        complete_time = pd.Series(data=[False]*168, index=pd.date_range(start='2023-01-01', periods=168, freq='H'))
        for _, row in group.iterrows():
            start_idx = complete_time.index.get_loc(row['start'], method='nearest')
            end_idx = complete_time.index.get_loc(row['end'], method='nearest')
            complete_time[start_idx:end_idx+1] = True
        
        full_week = (complete_time.resample('D').sum() == 24).all()
        full_day_coverage = (complete_time.groupby(complete_time.index.date).sum() == 24).all()
        result.append((id_, id_2, not (full_week and full_day_coverage)))
    
    result_df = pd.DataFrame(result, columns=['id', 'id_2', 'has_incorrect_timestamps'])
    result_df.set_index(['id', 'id_2'], inplace=True)
    
    return result_df['has_incorrect_timestamps']

df = pd.read_csv('dataset-1.csv')
result = verify_time_completeness(df)
print(result)

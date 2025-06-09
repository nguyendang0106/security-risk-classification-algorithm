# import requests
# import json

# data = {"data": [8.35298358e-02,5.50302144e-05,2.43137255e-01,1.20633645e-06,1.06906827e-03,
#                  8.02996772e-01,3.85135135e-02,1.06000000e-01,1.63934426e-02,2.13934201e-04]
# }

# url = "http://127.0.0.1:8888/predict/"

# data = json.dumps(data)

# respnse = requests.post(url, data)

# print(respnse.json())


import requests
import json


N_FEATURES_FOR_CLIENT = 67 
sample_features = [0.0] * N_FEATURES_FOR_CLIENT 

sample_features = [5.87000000e+04, 4.90000000e+01, 1.00000000e+00, 1.00000000e+00,
  6.00000000e+00, 6.00000000e+00, 6.00000000e+00, 6.00000000e+00,
  6.00000000e+00, 0.00000000e+00, 6.00000000e+00, 6.00000000e+00,
  6.00000000e+00, 0.00000000e+00, 2.44897959e+05, 4.08163265e+04,
  4.90000000e+01, 0.00000000e+00, 4.90000000e+01, 4.90000000e+01,
  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.00000000e+01,
  2.00000000e+01, 2.04081633e+04, 2.04081633e+04, 6.00000000e+00,
  6.00000000e+00, 6.00000000e+00, 0.00000000e+00, 0.00000000e+00,
  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
  1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00,
  9.00000000e+00, 6.00000000e+00, 6.00000000e+00, 1.00000000e+00,
  6.00000000e+00, 1.00000000e+00, 6.00000000e+00, 5.80000000e+01,
  1.64250000e+04, 0.00000000e+00, 2.00000000e+01, 0.00000000e+00,
  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
  0.00000000e+00, 0.00000000e+00, 0.00000000e+00]


if len(sample_features) != N_FEATURES_FOR_CLIENT:
    print(f"Warning: Sample data length ({len(sample_features)}) might not match expected features ({N_FEATURES_FOR_CLIENT}).")
    print("Please update 'sample_features' in client.py with a correct data sample.")

data_to_send = {"features": sample_features}

url = "http://127.0.0.1:8888/predict" 

try:
    response = requests.post(url, json=data_to_send) # Send data as JSON
    response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
    
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    print("Response content:", response.text)
except requests.exceptions.ConnectionError as conn_err:
    print(f"Connection error occurred: {conn_err}")
    print("Is the server running at the specified URL?")
except requests.exceptions.Timeout as timeout_err:
    print(f"Timeout error occurred: {timeout_err}")
except requests.exceptions.RequestException as req_err:
    print(f"An error occurred: {req_err}")
    print("Response content:", response.text if response else "No response")
except json.JSONDecodeError:
    print("Error decoding JSON response. Raw response:")
    print(response.text)

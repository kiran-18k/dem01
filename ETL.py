import requests
import pandas as pd

def extract_data_from_api(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

if __name__ == "__main__":
    api_url = "https://jsonplaceholder.typicode.com/posts"  # Sample API for demonstration
    data = extract_data_from_api(api_url)
    data.to_csv('extracted_data.csv', index=False)
    print("Data extraction complete. Saved to extracted_data.csv")
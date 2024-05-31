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
    



def transform_data(df):
    # Example transformation: Keep only certain columns and create a new column
    df_transformed = df[['userId', 'id', 'title']].copy()  # Keep only selected columns
    df_transformed['title_length'] = df_transformed['title'].apply(len)  # Add a new column for title length
    return df_transformed

if __name__ == "__main__":
    api_url = "https://jsonplaceholder.typicode.com/posts"  # Sample API for demonstration
    data = extract_data_from_api(api_url)
    
    # Transform the data
    transformed_data = transform_data(data)
    
    # Save the transformed data to a CSV file
    transformed_data.to_csv('transformed_data.csv', index=False)
    print("Data transformation complete. Saved to transformed_data.csv")
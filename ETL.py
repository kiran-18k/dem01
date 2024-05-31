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
    # Conflicting Change in FTE02 branch
    cols = ['userId', 'id']  # Changed the columns to 'userId' and 'id'
    df_transformed = df[cols].copy()  # Keep selected columns
    df_transformed['id_length'] = df_transformed['id'].apply(len)  # Add a new column for id length
    return df_transformed

def quality_check(df):
    # QC step to verify that the DataFrame is not empty after transformation
    if df.empty:
        raise ValueError("Transformed data is empty. Quality check failed.")
    print("Quality check passed. Transformed data is not empty.")

if __name__ == "__main__":
    api_url = "https://jsonplaceholder.typicode.com/posts"  # Sample API for demonstration
    data = extract_data_from_api(api_url)
    
    # Transform the data
    transformed_data = transform_data(data)
    
    # Perform quality check
    quality_check(transformed_data)
    
    # Save the transformed data to a CSV file
    transformed_data.to_csv('transformed_data.csv', index=False)
    print("Data transformation complete. Saved to transformed_data.csv")

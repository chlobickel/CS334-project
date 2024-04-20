import pandas as pd

def calculate_roadway_severity_scores(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Remove rows containing '(None)'
    df = df[~df.apply(lambda row: row.astype(str).str.contains(r'\(None\)').any(), axis=1)]

    # Check if necessary columns exist
    if 'Roadway (From Crash Report)' not in df.columns or 'KABCO Severity' not in df.columns:
        raise ValueError("The required columns are not in the dataframe")

    # Mapping textual severity descriptions to numeric values
    severity_mapping = {
        '(O) No Injury': 0,
        '(C) Possible Injury / Complaint': 1,
        '(B) Suspected Minor/Visible Injury': 2,
        '(A) Suspected Serious Injury': 3,
        '(K) Fatal Injury': 5
    }

    # Apply the mapping to the 'KABCO Severity' column
    df['Numeric Severity'] = df['KABCO Severity'].map(severity_mapping)

    # Filter out rows with 'Unknown' severity
    df = df[df['KABCO Severity'] != 'Unknown']

    # Group by 'Roadway (From Crash Report)' and calculate sum and count of 'Numeric Severity'
    grouped = df.groupby('Roadway (From Crash Report)')['Numeric Severity'].agg(['sum', 'count']).reset_index()
    grouped['Severity_Score'] = grouped['sum'] / grouped['count']

    # Create a dictionary to map Roadway to Severity Score
    severity_score_map = dict(zip(grouped['Roadway (From Crash Report)'], grouped['Severity_Score']))

    # Map the severity scores back to the original dataframe
    df['Roadway_Severity_Score'] = df['Roadway (From Crash Report)'].map(severity_score_map)

    # Save the updated dataframe back to csv
    df.to_csv('updated_cleaned_data.csv', index=False)

    print("Updated dataset with Roadway Severity Scores saved to 'updated_cleaned_data.csv'.")

# Example file path (replace 'cleaned_data.csv' with your actual file path)
file_path = 'cleaned_data.csv'
calculate_roadway_severity_scores(file_path)

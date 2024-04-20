import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Define a function to process each chunk
def process_chunk(chunk, chunk_index):
    print(f"Processing chunk {chunk_index+1}...")
    chunk.columns = chunk.columns.str.strip()

    # Convert 'Date and Time' to datetime
    chunk['Date and Time'] = pd.to_datetime(chunk['Date and Time'], format='%m/%d/%Y %I:%M %p', errors='coerce')
    chunk['Season'] = chunk['Date and Time'].dt.month.apply(lambda month: 'Spring' if month in (3, 4, 5) else 'Summer' if month in (6, 7, 8) else 'Fall' if month in (9, 10, 11) else 'Winter')
    chunk['Weekday'] = chunk['Date and Time'].dt.day_name()
    chunk['Time Category'] = chunk['Date and Time'].dt.hour.apply(
        lambda hour: 'Midnight to 3 AM' if 0 <= hour < 3 else '3 AM to 6 AM' if 3 <= hour < 6 else '6 AM to 9 AM' if 6 <= hour < 9 else '9 AM to Noon' if 9 <= hour < 12 else 'Noon to 3 PM' if 12 <= hour < 15 else '3 PM to 6 PM' if 15 <= hour < 18 else '6 PM to 9 PM' if 18 <= hour < 21 else '9 PM to Midnight'
    )
    chunk.drop('Date and Time', inplace=True, axis=1)

    # Remove 'Roadway (From Crash Report)' and 'KABCO Severity' columns
    chunk.drop(['Roadway (From Crash Report)', 'KABCO Severity'], inplace=True, axis=1)

    # Specify all categorical columns to encode
    categorical_cols = ['Season', 'Weekday', 'Time Category', 'Weather Conditions (Crash Level)', 'Surface Condition (Crash Level)', 'Light Conditions (Crash Level)']
    encoder = OneHotEncoder()
    encoded_columns = encoder.fit_transform(chunk[categorical_cols]).toarray()  # Convert to dense array
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate with one-hot encoded columns
    chunk = chunk.drop(columns=categorical_cols)
    chunk = pd.concat([chunk.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    return chunk

# Read data in chunks
chunks = pd.read_csv('updated_cleaned_data.csv', chunksize=5000)
processed_chunks = []
for index, chunk in enumerate(chunks):
    processed_chunk = process_chunk(chunk, index)
    processed_chunks.append(processed_chunk)
    print(f"Chunk {index+1} processed successfully.")

# Concatenate all processed chunks
df_cleaned = pd.concat(processed_chunks, ignore_index=True)

# Save the final DataFrame
df_cleaned.to_csv('processed_data.csv', index=False)
print("All data processed and saved.")

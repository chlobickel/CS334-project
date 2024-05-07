import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
data = pd.read_csv('Raw Data.csv', delimiter=',', quotechar='"')

# Specify the date and time format
date_format = "%m/%d/%Y %I:%M %p"  # Adjust this format to match your data

# Convert 'Date and Time' to a numerical format using the specified date format
data['Time Numeric'] = pd.to_datetime(data['Date and Time'], format=date_format).dt.hour * 60 + pd.to_datetime(data['Date and Time'], format=date_format).dt.minute

# Initialize LabelEncoder for categorical columns
label_encoder = LabelEncoder()

# Exact column names including spaces where necessary
categorical_columns = [
    'Agency Name (Crash Level) ', 'Area: County', 'Area: City', 'Roadway (From Crash Report)',
    'Intersection Name (from Crash Report)', 'KABCO Severity', 'Manner of Collision (Crash Level) ',
    '# of Fatalities per Crash', '# Serious Injuries', '# Visible Injuries', '# Complaint Injuries',
    '# of Vehicles per crash ', 'Weather Conditions (Crash Level)', 'Surface Condition (Crash Level) ',
    'Light Conditions (Crash Level)'
]

# Encode categorical variables
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column].astype(str))

# Calculate the correlation matrix
corr_matrix = data.select_dtypes(include=[np.number]).corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

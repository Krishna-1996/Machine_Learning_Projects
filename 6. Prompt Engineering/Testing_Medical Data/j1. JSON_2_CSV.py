import json
import pandas as pd

# Load the JSON data from the corono2.json file
with open('corono2.json', 'r') as file:
    data = json.load(file)

# Define a function to process the data into a flat format
def process_data(data):
    records = []
    for item in data:
        # Prepare a record for each entry in the JSON data
        record = {
            "id": item.get('id'),
            "tag_id": item.get('tag_id'),
            "tag_name": item.get('tag_name'),
            "value": item.get('value'),
            "start": item.get('start'),
            "end": item.get('end'),
            "example_id": item.get('example_id')
        }
        records.append(record)
    
    return pd.DataFrame(records)

# Process data into a flat format
df = process_data(data)

# Save as different CSV formats:

# 1. Save CSV with all columns
df.to_csv('corono2_full.csv', index=False)

# 2. Save CSV with only the columns relevant to the tag information
df_tag_only = df[['id', 'tag_id', 'tag_name', 'value']]
df_tag_only.to_csv('corono2_tags.csv', index=False)

# 3. Save CSV with only tag name and the associated values
df_value_only = df[['tag_name', 'value']]
df_value_only.to_csv('corono2_values.csv', index=False)

print("Data has been processed and saved as CSV.")

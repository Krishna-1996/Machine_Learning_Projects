#%%

import json
import pandas as pd

# Load the JSON data from the corono2.json file
with open('Corona2.json', 'r') as file:
    data = json.load(file)

# Check the first few entries of the 'examples' key to understand the structure
print(data['examples'][0].keys())

#%%
# Prepare the training data from the 'examples' key
training_data = []
for example in data['examples']:
    content = example['content']  # The text content
    annotations = example['annotations']  # The list of annotations (tags)
    
    # Create records for each annotation
    for annotation in annotations:
        record = {
            'text': content,  # Text from the 'content' field
            'tag_name': annotation['tag_name'],  # The tag (entity type)
            'start': annotation['start'],  # The start position of the entity
            'end': annotation['end'],  # The end position of the entity
        }
        training_data.append(record)

# Convert the processed data into a pandas DataFrame
df = pd.DataFrame(training_data)


#%%
# Save as different CSV formats:

# 1. Save CSV with all columns (text, tag_name, start, end)
df.to_csv('Corona2_full.csv', index=False)

# 2. Save CSV with only the columns relevant to the tag information (tag_name and text)
df_tag_only = df[['text', 'tag_name']]
df_tag_only.to_csv('Corona2_tags.csv', index=False)

# 3. Save CSV with only the tag name and the associated positions (start, end)
df_position_only = df[['tag_name', 'start', 'end']]
df_position_only.to_csv('Corona2_positions.csv', index=False)

print("Data has been processed and saved as CSV.")

# %%

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Load Excel data
excel_file_path = 'data.xlsx'  # Path to your Excel file
df = pd.read_excel(excel_file_path)

# Initialize the Selenium WebDriver (example for Chrome)
driver = webdriver.Chrome(executable_path='path/to/chromedriver')

# Open the AWS form
aws_form_url = 'https://innovate-migrate-modernize-build-apj.virtual.awsevents.com/register?trk=95f26c9a-a33e-4fe4-99fd-c80417f6b8d3&trkcampaign=aws-innovate'
driver.get(aws_form_url)

# Give some time for the page to load
time.sleep(10)

# Function to fill the form based on row data
def fill_form(row):
    # Example: Locate a form field and fill it with data from the Excel sheet
    name_field = driver.find_element(By.NAME, 'your_name_field')  # Use the actual 'name' attribute of the input field
    name_field.clear()  # Clear previous value
    name_field.send_keys(row['Name'])  # Assume 'Name' is a column in your Excel file
    
    email_field = driver.find_element(By.NAME, 'your_email_field')
    email_field.clear()
    email_field.send_keys(row['Email'])
    
    # Repeat for other fields in the form
    # ...

# Loop through each row in the Excel sheet
for index, row in df.iterrows():
    # Fill the form with current row data
    fill_form(row)
    
    print(f"Form filled with data from row {index + 1}. Please submit manually.")
    
    # Wait for manual submission by user
    input("Press Enter after you've manually submitted the form...")
    
    # Refresh the page to reset the form for the next row of data
    driver.refresh()
    
    # Give some time for the page to reload before the next iteration
    time.sleep(3)

# Close the browser when done
driver.quit()

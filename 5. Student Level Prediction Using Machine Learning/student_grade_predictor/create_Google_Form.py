import os
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Authenticate with Google
creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"])

# Create a Google Sheets client
sheets_service = build('sheets', 'v4', credentials=creds)
drive_service = build('drive', 'v3', credentials=creds)

# Create a Google Sheet to store the data (this is just for tracking the data)
spreadsheet = {
    'properties': {'title': 'Student Details Form Data'}
}

try:
    sheet = sheets_service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
    sheet_id = sheet['spreadsheetId']
    print(f"Created spreadsheet with ID: {sheet_id}")
except HttpError as err:
    print(f"An error occurred: {err}")
    sheet_id = None

# Google Apps Script to create the Google Form
app_script = """
function createForm() {
  var form = FormApp.create('Student Details Form');

  // Section 1: Student Details
  var section1 = form.addSectionHeaderItem().setTitle('Section 1: Student Details');

  form.addMultipleChoiceItem()
    .setTitle('What is the student\'s gender?')
    .setChoiceValues(['Female', 'Male']);

  form.addMultipleChoiceItem()
    .setTitle('What is the student\'s age?')
    .setChoiceValues(['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']);

  form.addMultipleChoiceItem()
    .setTitle('What is the student\'s current year (2017-2018)?')
    .setChoiceValues(['FS2', 'Grade 1', 'Grade 10', 'Grade 11', 'Grade 12', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'KG1', 'KG2']);

  form.addMultipleChoiceItem()
    .setTitle('What is the proposed year/grade for 2018-2019?')
    .setChoiceValues(['Grade 1', 'Grade 10', 'Grade 11', 'Grade 12', 'Grade 13', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'KG2']);

  form.addMultipleChoiceItem()
    .setTitle('What was the previous curriculum?')
    .setChoiceValues(['American', 'British']);

  form.addMultipleChoiceItem()
    .setTitle('What is the student\'s current school?')
    .setChoiceValues(['School 1', 'School 2']);

  form.addMultipleChoiceItem()
    .setTitle('What is the student\'s current curriculum?')
    .setChoiceValues(['American', 'British']);

  form.addMultipleChoiceItem()
    .setTitle('What grade system was followed in the previous year?')
    .setChoiceValues(['Grade System', 'Year System']);

  // Section 2: Student Exam Scores
  var section2 = form.addSectionHeaderItem().setTitle('Section 2: Student Exam Scores');

  form.addTextItem()
    .setTitle('Student\'s marks in English 2020 Term 3');

  form.addTextItem()
    .setTitle('Student\'s marks in Science 2020 Term 3');

  form.addTextItem()
    .setTitle('Student\'s marks in Math 2020 Term 3');

  // Continue adding other subjects and terms...

  Logger.log('Form URL: ' + form.getEditUrl());
}
"""

# Write the script to a Google Apps Script file (this could be automated via API but here is the manual process)
try:
    file_metadata = {
        'name': 'CreateStudentForm',
        'mimeType': 'application/vnd.google-apps.script'
    }
    media = MediaIoBaseDownload(app_script, 'application/vnd.google-apps.script')
    file = drive_service.files().create(body=file_metadata, media_body=media).execute()
    print(f"Google Apps Script created: {file['id']}")
except HttpError as err:
    print(f"An error occurred: {err}")

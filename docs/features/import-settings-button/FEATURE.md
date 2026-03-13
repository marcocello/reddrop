# Feature: Import settings from JSON

## Scenario: import a valid settings file
Given the user is on the Settings page  
When the user clicks "Import settings" and selects a valid JSON file with `reddit` and `openrouter` sections  
Then the settings form fields are populated from the file  
And the UI shows a success message telling the user to save

## Scenario: reject invalid settings file
Given the user is on the Settings page  
When the user imports a JSON file that is malformed or missing required sections  
Then the UI shows an error message  
And existing form values are not replaced with invalid data

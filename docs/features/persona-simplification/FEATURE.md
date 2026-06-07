# Feature: Persona simplification and OpenRouter-only settings

## Scenario: create or edit a persona with a single combined profile field
Given the user opens the Personas create or edit form  
When the user configures a persona profile  
Then the form collects only `name` and `objective`  
And the form does not collect a separate `description` field  
And the objective input label is "Objective, tone & style"  
And the objective input placeholder is:
"""
## Objectives
......

# Tone & Style
.....
"""
And the create persona modal body uses a taller scroll area (`max-h-[80vh]`)

## Scenario: persona API persists the simplified persona model
Given the persona API receives a persona payload  
When the payload includes `name` and `objective`  
Then the API saves and returns the same fields  
And it does not require `description` or `tone_style`

## Scenario: settings are OpenRouter-only
Given the user opens runtime settings  
When settings are loaded or saved  
Then the settings payload contains only the `openrouter` section  
And the Settings UI does not show reply system prompt configuration

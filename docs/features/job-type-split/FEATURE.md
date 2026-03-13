# Feature: Split jobs by type

## Scenario: search job runs search and rank
Given a job with type `search`  
When a run starts for that job  
Then it executes Reddit discovery with ranking  
And it does not execute reply generation in the same run

## Scenario: reply job runs from search artifact
Given a job with type `reply` linked to a search job  
When a run starts for the reply job  
Then it loads threads from the linked search job artifact  
And it generates replies for those threads

## Scenario: reply job creation validation
Given the user creates a reply job  
When `source_job_id` or personas are missing  
Then the API rejects the request with a validation error

## Scenario: reply job persona is selected from dropdown
Given the user configures a reply job  
When the user picks persona  
Then the persona is selected from a dropdown menu

## Scenario: reply job name is prefixed from source search job
Given the user selects job type `reply`  
And the user links a source search job  
When the user saves the job  
Then the name is composed as `<editable-prefix>reply_<source-search-job>`  
And the UI shows `Name` as two adjacent fields (editable prefix + locked suffix)
And reply jobs do not keep a topic value (shown as `N/A` in job views)
And reply jobs allow a configurable minimum similarity score

## Scenario: discovery logs keep all five steps
Given discovery runs even with zero candidate threads  
When ranking has no candidates  
Then `step 5/5` is still printed in logs

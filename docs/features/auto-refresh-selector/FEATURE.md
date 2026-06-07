# Feature: Auto-refresh selector for Jobs and Threads

## Scenario: Jobs and Threads expose selectable auto-refresh frequencies
Given the Jobs or Threads page is open
When the header controls are rendered
Then the UI shows a compact split control:
And left side is a manual refresh action (refresh icon)
And right side is an auto-refresh dropdown trigger (chevron + current value)
And the menu options are `Auto-refresh off`, `Every 30s`, `Every 1m`, `Every 5m`, `Every 15m`
And the default selected value is `Every 30s`

## Scenario: Polling interval follows selected refresh frequency
Given the user changes auto-refresh frequency in Jobs or Threads
When the page schedules polling
Then the polling timer interval uses `autoRefreshSeconds * 1000`

## Scenario: Auto-refresh text label is removed
Given the page header has refresh controls
When the UI is rendered
Then no `Auto-refresh:` label text is shown
And a manual refresh action remains available inside the split control

## Scenario: Jobs header keeps create and Threads header is refresh-only
Given the Jobs and Threads pages are open
When header actions are rendered
Then the Jobs page shows a `Create job` action
And the split refresh control appears before the `Create job` action in Jobs
And the Threads page does not show a `Create job` action
And both pages keep manual and auto-refresh actions in the split control

## Scenario: Create-action pages use refresh-first ordering
Given the Personas, Auths, and Search Queries pages are open
When header actions are rendered
Then each page shows a `Refresh` action first
And each page shows a `Create ...` action second
And each `Create ...` button includes a plus icon

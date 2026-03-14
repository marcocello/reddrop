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

## Scenario: Jobs and Threads headers use consistent actions
Given the Jobs and Threads pages are open
When header actions are rendered
Then both pages show a `Create job` action
And the `Create job` action appears before the split refresh control
And both pages keep manual and auto-refresh actions in the split control

## Scenario: Personas header does not include a refresh action
Given the Personas page is open
When header actions are rendered
Then only `Create persona` is shown
And no manual refresh button is rendered

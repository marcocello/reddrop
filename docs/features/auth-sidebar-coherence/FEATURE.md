# Feature: Auth sidebar coherence

## Scenario: Auth details sidebar matches the app sidebar layout pattern
Given the Auths page is open
When an auth row is selected
Then the right sidebar uses the same structural layout pattern as Jobs and Personas:
And `SheetContent` uses edge-to-edge content (`p-0`) with a constrained width
And the sidebar content container is a full-height column layout
And the header uses padded spacing consistent with other sidebars
And actions are shown in a sticky top action row above scrollable content

## Scenario: Auth sidebar actions remain available in the sticky action row
Given the selected auth is `linkedin`
When the auth sidebar is opened
Then the sticky action row includes `Run LinkedIn login`
And the sticky action row includes `Save changes`
And `Delete` remains available for selected auths

## Scenario: Auth and Settings pages expose top-level page titles
Given the authenticated app navigation is available
When the Auth page is rendered
Then a top-level heading `Auth` is visible like other app pages
When the Settings page is rendered
Then a top-level heading `Settings` is visible like other app pages

## Scenario: Auth page main vertical spacing matches Personas page rhythm
Given the Auth page and Personas page are rendered
When comparing their main content layout structure
Then both pages use the same main vertical spacing pattern (`gap-4 sm:gap-6`)
And Auth keeps the same heading, toolbar, and table section stacking pattern as Personas

# Auth Objects for Reddit and LinkedIn

## Acceptance Scenarios
Feature: Auth object management and social login bootstrap
  In order to run Reddit and LinkedIn jobs with the correct account context
  As an operator configuring automation jobs
  I want to manage named auth objects and trigger LinkedIn session login from the Auths flow

  Scenario: Create a Reddit auth object
    Given I open the Auths page
    When I create a new auth object with social "reddit"
    And I provide Reddit username and password credentials
    Then the auth object is saved
    And it is listed by its name in the Auths page
    And it is selectable by name in the Jobs form

  Scenario: Create a LinkedIn auth object triggers browser login bootstrap
    Given I open the Auths page
    When I create a new auth object with social "linkedin"
    Then a LinkedIn login browser flow is triggered automatically
    And the browser closes when the login window wait completes
    And the auth object is saved only when login completed successfully
    And its ID is generated automatically as a UUID
    And its user data directory is derived from the auth name and ID
    And the API reports the login bootstrap execution result

  Scenario: LinkedIn auth creation is rejected when login is not completed
    Given I open the Auths page
    And I create a new auth object with social "linkedin"
    When the login flow ends without an authenticated LinkedIn session
    Then the auth object is not persisted
    And the UI reports that LinkedIn login was not completed

  Scenario: LinkedIn auth creation does not request username/password fields
    Given I choose social "linkedin" in the Auths create form
    When the form is rendered
    Then username and password input fields are not shown
    And the form asks only for browser-session settings such as wait seconds

  Scenario Outline: Jobs use auth names for selection and display
    Given a "<social>" auth object exists
    And I open the Jobs create form
    When I select the auth profile for a job
    Then the selected auth is shown by its name
    And the Jobs table and details show the auth name without extra ID noise

    Examples:
      | social   |
      | reddit   |
      | linkedin |

  Scenario: Sidebar navigation groups are aligned with the workflow
    Given I open the authenticated dashboard
    When the sidebar is rendered
    Then it shows a "Workflows" group containing Jobs, Threads, and Personas
    And it shows a "Configuration" group containing Auths and Settings

  Scenario: LinkedIn login bootstrap runs from API without module resolution errors
    Given a LinkedIn auth object exists
    When I trigger LinkedIn login from that auth object
    Then the backend executes the login bootstrap command successfully
    And the command resolves the local "app.linkedin_scraper" module in the project runtime
    And the browser is opened in private mode for the login flow

## Constraints
- Settings page remains LLM-only; social credentials are managed in Auths.
- LinkedIn login bootstrap must open a browser session and save profile state in the configured user data directory.
- Playwright dependency must be available in backend runtime for LinkedIn login bootstrap.

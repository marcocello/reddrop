# Search Query Resource

## Acceptance Scenarios
Feature: Search query creation is a separate phase from search-job creation
  In order to lock deterministic search inputs before runtime
  As a user managing social discovery
  I want to create and save search queries, then bind jobs to them

  Scenario: Generate a draft search query with platform-aware planning
    Given a social platform and a topic
    When I generate a search query
    Then a draft search query is persisted with a platform-specific plan

  Scenario: Save a search query before creating a search job
    Given a draft search query
    When I save it
    Then its status becomes saved and version is set to 1

  Scenario: Search jobs can bind to saved search queries only
    Given a saved search query
    When I create a search job with that search query id
    Then the job stores search_query_id and search_query_version
    And runtime search resolves topic and plan from that saved search-query reference

  Scenario: Search jobs require search_query_id
    Given no search query id is provided
    When I create a search job
    Then the API rejects the request

  Scenario: Draft search queries cannot be used to create search jobs
    Given a draft search query
    When I create a search job with that search query id
    Then the API rejects the request

  Scenario: Draft detail header actions remain visible on smaller sheet widths
    Given a draft search query is opened in the right-side details sheet
    When the row contains id/status badges and action buttons
    Then the header layout wraps actions instead of clipping buttons
    And the search query id badge truncates long ids to preserve action space

  Scenario: Draft sidebar prioritizes search-plan editing space
    Given a draft search query is opened in the right-side details sheet
    When metadata and search-plan fields are rendered
    Then the source topic field uses a compact height
    And search-plan fields use larger text areas for easier editing

  Scenario: Search-job create form only selects a saved search query
    Given I am creating a Search job
    When the form is rendered
    Then it shows a saved search-query selector
    And it does not show a topic field
    And it does not show an inline generate-search-query action
    And it does not show inline topics/subreddits/queries editors

  Scenario: Runtime logs report effective locked search inputs
    Given a Search job references a saved search query
    When runtime starts discovery
    Then LinkedIn logs include the effective query and search-query reference
    And Reddit logs include effective topics, subreddits, and queries from the locked plan

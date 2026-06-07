# Search Plan Locking

## Acceptance Scenarios
Feature: Search jobs use pre-generated locked search plans
  In order to keep deterministic search behavior across runs
  As a user creating search jobs
  I want to generate and edit a search plan before job creation and then save it

  Scenario: Reddit search plan generation returns editable topics, subreddits, and queries
    Given a topic
    When I request search plan generation
    Then the API returns a plan with topics, subreddits, and queries

  Scenario: LinkedIn search plan generation returns an editable query
    Given a topic
    When I request search plan generation
    Then the API returns a plan with a LinkedIn query generated via LLM
    And if LLM is unavailable the topic is used as fallback query

  Scenario: Creating a search job stores saved search-query reference only
    Given I generated and edited a search plan
    When I create a search job
    Then the persisted job includes search_query_id and search_query_version
    And the persisted job does not include topic or search_plan snapshots
    And the job cannot be edited after creation

  Scenario: Runtime search uses only the locked search plan
    Given a search job references a saved search query
    When the job runs
    Then runtime resolves topic and plan from that saved search query
    And discovery uses only those locked plan values for subreddit/query selection
    And it does not regenerate subreddit/query inputs at runtime

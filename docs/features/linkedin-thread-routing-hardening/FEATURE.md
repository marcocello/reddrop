# LinkedIn Thread Routing Hardening

## Problem
LinkedIn thread identifiers could include `/` characters or URN formats that were not handled consistently between discovery and send flows. This caused thread-level send/reply endpoints to fail for valid LinkedIn posts and produced malformed post URLs for send operations.

## Behavior Contract

```gherkin
Feature: LinkedIn thread identity routing and URL reconstruction

  Scenario: Send endpoint accepts LinkedIn thread IDs that contain slashes
    Given a LinkedIn search artifact stores a conversation thread_id as a full LinkedIn post URL
    When the thread send endpoint is called for that thread_id
    Then the request is routed successfully
    And the LinkedIn posting service receives the exact stored thread_id

  Scenario: URN thread IDs map to canonical LinkedIn feed URLs
    Given a LinkedIn thread_id in URN format
    When the posting URL is resolved
    Then the resolved URL is "https://www.linkedin.com/feed/update/<urn>/"

  Scenario: LinkedIn send falls back to stored URL when URN send fails
    Given a LinkedIn conversation stores thread_id as an URN and url as a canonical feed URL
    And the first send attempt using thread_id fails
    When the thread send endpoint is called for that conversation
    Then the endpoint retries send using the stored URL
    And the thread is marked as sent when the URL attempt succeeds

  Scenario: Job names are unique case-insensitively
    Given a job named "Growth" already exists
    When a new job named "growth" is created
    Then the API rejects the create request with conflict
```

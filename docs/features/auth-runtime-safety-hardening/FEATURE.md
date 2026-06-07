# Auth Runtime Safety Hardening

## Problem
Runtime flows still depended on global `os.environ` mutation for auth credentials, auth/persona catalog writes were not lock-safe atomic writes, deleting a search job could orphan linked reply jobs, and LinkedIn post-send success detection was too optimistic.

## Behavior Contract

```gherkin
Feature: Runtime auth, persistence, and LinkedIn send safety

  Scenario: Search and send runtime service construction uses job auth context
    Given two jobs can run with different auth profiles
    When a runtime path prepares auth for a job
    Then platform service instances receive auth explicitly for that run
    And process-global auth environment variables are not overwritten by runtime auth prep

  Scenario: Auth and persona stores persist safely under concurrent writers
    Given auth and persona catalog files can be written by multiple backend processes
    When auths or personas are saved
    Then writes use file locking and atomic replace semantics
    And direct truncating writes to target JSON files are not used

  Scenario: Source search jobs with linked reply jobs require explicit cascade
    Given a search job has one or more reply jobs linked by source_job_id
    When the search job delete endpoint is called
    Then the API rejects deletion with conflict unless cascade is explicitly requested
    And when cascade is requested, linked reply jobs are deleted together with the source search job

  Scenario: LinkedIn send only succeeds with verifiable submission signals
    Given a LinkedIn post reply is submitted from the browser UI
    When the submit action is executed
    Then success requires a verifiable signal: comment-submission voyager POST 2xx acknowledgement, cleared comment editor, or visible posted comment text
    And submit click is retried once before reporting failure
    And auth wall or challenge pages are treated as unauthenticated in send flows
    And unverified submit attempts are reported as failed
```

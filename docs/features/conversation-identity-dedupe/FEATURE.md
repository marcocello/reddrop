# Conversation Identity And URL Dedupe

## Problem
Conversation rows in search artifacts must use stable UUID identity derived from post URL, and merge behavior must never persist duplicate rows that point to the same canonical URL.

## Behavior Contract

```gherkin
Feature: Conversation identity and URL dedupe

  Scenario: Conversation rows always have deterministic URL-based UUID identity
    Given search/reply/send flows persist conversations in search artifacts
    When a conversation row is created or merged
    Then the row includes an `id` value
    And `id` is a UUID derived from the canonical post URL

  Scenario: Conversations are deduped by URL
    Given a search artifact already contains a conversation URL
    When incoming discovery results contain another conversation with the same URL
    Then only one conversation row for that URL is persisted
    And both rows resolve to the same deterministic conversation UUID
    And existing reply/send state is preserved

  Scenario: Threads API exposes conversation identity
    Given threads are listed from persisted artifacts
    When `/threads` returns rows
    Then each row includes `conversation_id`
    And `conversation_id` is the deterministic UUID for that row URL

  Scenario: Rerun initialization does not wipe existing conversations on failure
    Given a search artifact already contains conversations for a job
    When a rerun starts and discovery fails before new results are saved
    Then the previously persisted conversations remain in the artifact

  Scenario: Threads listing dedupes the same URL discovered by different jobs
    Given two search jobs persist conversations with the same canonical URL
    When `/threads` is listed without job filters
    Then the duplicated URL appears once
    And reply/comment state from the duplicate rows is preserved on the kept row

  Scenario: LinkedIn activity and ugcPost URL variants dedupe as one post
    Given LinkedIn conversations include both `/feed/update/urn:li:activity:*` and `/feed/update/urn:li:ugcPost:*` variants for the same post content
    When conversations are persisted and `/threads` is listed
    Then those variants are treated as one conversation
    And reply/comment state from duplicate variants is preserved

  Scenario: Search persistence maintains a single consolidated conversation file
    Given multiple search artifacts exist for different jobs
    When search persistence writes updates
    Then `backend/.reddrop/runs/conversation.json` is refreshed
    And it contains all persisted job artifacts and flattened conversations
    And it does not persist `topic` fields

  Scenario: Search persistence avoids per-artifact lock-file proliferation
    Given historical per-artifact lock files may exist
    When search persistence initializes and writes artifacts
    Then legacy `search_*.json.lock` files are cleaned up
    And locking uses a single shared `.search_storage.lock` file

  Scenario: Search artifact files do not persist topic
    Given search runtime resolves topic from saved search paths
    When `search_*.json` artifacts are written
    Then the artifact payload omits `topic`
```

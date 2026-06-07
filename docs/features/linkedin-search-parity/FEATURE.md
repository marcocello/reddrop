# LinkedIn Search Parity

## Problem
`LinkedinService.discover_relevant_threads` still contained legacy bridge wiring to an external `linkedin_scraper` repository and optional fallback flags.

That stale path must be removed entirely so runtime discovery uses a single in-process implementation.

## Behavior Contract

```gherkin
Feature: LinkedIn discovery uses native in-process extraction

  Scenario: Runtime LinkedIn discovery uses native Playwright extraction only
    Given a LinkedIn search job with query, threads_limit, and scroll_rounds
    When discovery runs
    Then the service discovers posts using the in-process Playwright flow
    And it does not execute an external scraper subprocess
    And rows are limited by `threads_limit`

  Scenario: Missing Playwright dependency fails explicitly
    Given Playwright is not available in runtime
    When discovery runs
    Then discovery fails with a runtime error that explains Playwright is required

  Scenario: LinkedIn discovery supports recency filtering and ordering
    Given LinkedIn post cards include relative age labels and engagement counts
    When discovery runs with `time_filter` and `sort_by`
    Then rows outside the time filter are excluded
    And rows with unknown timestamp are included by default for bounded filters to preserve recall
    And bounded filters can be switched to strict mode via runtime configuration
    And rows can be ordered by `recency` or `engagement`
    And each row includes parsed `created_utc`, `score`, and `num_comments` when available

  Scenario: LinkedIn discovery computes semantic similarity from search topic
    Given a LinkedIn search run with a search-query topic context
    When discovery returns post rows
    Then each row includes `semantic_similarity` scored against the search topic
    And if semantic scoring dependencies are unavailable, discovery continues with `semantic_similarity=0.0`

  Scenario: API/runtime propagation keeps LinkedIn search controls aligned
    Given a LinkedIn search job is created with `time_filter`, `sort_by`, and `scroll_rounds`
    When the job is executed via `/search`
    Then runtime discovery receives the same values without dropping or remapping them

  Scenario: Create-job LinkedIn auth selector prefers username labels
    Given LinkedIn auth profiles include resolved usernames
    When the create-job form renders auth profile options
    Then LinkedIn options are labeled with username via shared display-name logic

  Scenario: Discovery URL matches LinkedIn faceted search parameters
    Given a query like `pluto AND pizza`
    When sort is recency
    Then discovery uses `origin=GLOBAL_SEARCH_HEADER` and `sortBy=["date_posted"]`
    When sort is relevance with past month
    Then discovery uses `origin=FACETED_SEARCH`, `sortBy=["relevance"]`, and `datePosted=["past-month"]`

  Scenario: Discovery keeps only real post permalinks
    Given content search markup includes both company `/company/.../posts/` links and post permalinks
    When discovery extracts conversations
    Then company pages are excluded
    And only activity/update post URLs are persisted

  Scenario: Discovery mirrors two-phase LinkedIn scraping with bounded runtime
    Given the search page yields seed post cards and valid post permalinks
    When discovery runs in headless Playwright mode
    Then phase 1 extracts seed posts from post-card markup
    And phase 1 waits for feed selectors before and during scroll rounds
    And scrolling uses wheel-style navigation with JavaScript fallback
    And phase 1 can fall back to strict permalink-anchor extraction when card selectors are sparse
    And phase 2 visits each discovered post permalink to enrich title/text/engagement
    And phase 2 waits for detail selectors and expands comment threads before extraction
    And phase 2 detail visits are capped by the configurable detail-enrichment limit and `threads_limit`
    And runtime accepts tunable LinkedIn scraping controls for selector waits and comment expansion

  Scenario: Authwall pages fail explicitly during search and detail phases
    Given LinkedIn can return authwall HTML without direct login redirects
    When discovery loads search or detail pages
    Then authwall content is detected
    And discovery fails with a clear re-login instruction

  Scenario: Scroll fallback and seed scanning improve recall in headless mode
    Given LinkedIn wheel scrolling may not always move the page in headless execution
    When discovery performs scroll rounds
    Then it falls back to JavaScript scrolling if wheel scrolling does not advance position
    And JavaScript fallback scroll targets LinkedIn's active results container (`main#workspace`) before document root
    And phase-1 seed extraction scans a bounded set larger than `threads_limit`

  Scenario: Seed discovery accumulates rows across scroll rounds
    Given LinkedIn content search may virtualize and replace visible post cards while scrolling
    When discovery runs multiple scroll rounds
    Then phase-1 seed extraction merges unique post rows from each round
    And final candidates are not limited to only the last visible cards
    And scrolling stops early after repeated rounds with no new unique rows

  Scenario: Generic LinkedIn `/posts/<slug>` permalinks are retained
    Given LinkedIn search results include post permalinks without `activity-<id>` suffixes
    When discovery canonicalizes extracted URLs
    Then valid `/posts/<slug>` permalinks are kept
    And company listing URLs like `/company/<name>/posts/` are still excluded

  Scenario: API and UI expose LinkedIn advanced scraping controls
    Given a user creates a LinkedIn search job
    When they configure selector wait timeout, detail wait timeout, and max comment expand clicks
    Then the job payload persists those values
    And runtime discovery receives the same values
    And users can tune how many seed posts receive detail-page enrichment

  Scenario: Detail enrichment is bounded and prioritized
    Given phase-1 seed extraction may discover many candidate post URLs
    When discovery runs
    Then phase-2 detail enrichment only visits a bounded top subset
    And the default detail enrichment cap is 12 URLs per run
    And the cap can be overridden per LinkedIn search job
    And rows missing timestamps or richer body fields are prioritized first for detail enrichment

  Scenario: Search query generation plans strict and broad LinkedIn queries
    Given a user provides a LinkedIn search topic
    When a search plan is generated
    Then it stores five possible LinkedIn content-search queries
    And the Search Queries UI displays the five LinkedIn queries as separate editable inputs
    And draft LinkedIn search queries can add or remove query inputs before saving
    And the first query remains available as the primary query for compatibility
    And invalid URL-style, site-operator, or non-content-search entries are discarded
    And sparse or invalid LLM output is supplemented with deterministic LinkedIn keyword queries

  Scenario: Search runs only saved LinkedIn plan queries
    Given the initial strict LinkedIn query may return sparse rows
    And the saved LinkedIn search plan contains broader query alternatives
    When discovery runs
    Then it searches only the configured saved queries
    And it does not append the raw topic as an extra fallback query
    And results are deduplicated across query passes
    And query-pass counters are logged to explain where rows were dropped

  Scenario: Search plans can fan out across multiple LinkedIn queries
    Given a saved LinkedIn search plan contains more than one query
    When discovery runs
    Then each configured query is searched in plan order
    And duplicate posts across queries are persisted once

  Scenario: Detail auth failures fail the run explicitly
    Given a LinkedIn detail page returns an auth wall
    When phase-2 enrichment opens that post
    Then discovery fails with a re-login instruction instead of silently returning sparse rows

  Scenario: Structured engagement counts are retained
    Given LinkedIn result markup exposes reaction and comment counts outside the post body
    When discovery extracts conversations
    Then persisted rows use those structured counts for score and comment metrics

  Scenario: LinkedIn profile sessions are single-writer at runtime
    Given two runtime actions use the same LinkedIn persistent profile directory
    When either action opens Playwright with that profile
    Then the service serializes access by profile directory to avoid concurrent profile use

  Scenario: Discovery recovers post URNs from encoded LinkedIn hydration data
    Given LinkedIn markup exposes post URNs as URL-encoded or JavaScript-escaped values
    When discovery canonicalizes extracted rows
    Then escaped activity URNs are decoded into canonical feed update URLs
    And the conversations are retained instead of being dropped as invalid URLs

  Scenario: Discovery exposes real published timestamp when LinkedIn provides datetime
    Given a LinkedIn post row includes an absolute `time[datetime]` value
    When discovery normalizes the row
    Then `published_at` is emitted in ISO-8601 UTC form
    And `created_utc` is derived from that absolute timestamp
    And when `published_at` is missing, only dedicated post-time labels are parsed
    And when only an imprecise relative label is available (for example `5d`), `published_at` is saved as date-only (`YYYY-MM-DD`)
    And body text is never used as a timestamp source

  Scenario: LinkedIn titles are bounded for conversation-list readability
    Given a discovered LinkedIn post title exceeds 90 characters
    When discovery maps the row for persistence
    Then `title` is truncated to 90 characters
    And `title` is derived from post body text when available (not author/company labels)

  Scenario: Conversation created values preserve time precision
    Given a conversation row includes `published_at` or `created_utc` with time-of-day
    When the conversations table renders the `Created` column
    Then the value is shown as UTC date-time (`YYYY-MM-DD HH:MM`)
    And if no valid timestamp exists, `ND` is shown
```

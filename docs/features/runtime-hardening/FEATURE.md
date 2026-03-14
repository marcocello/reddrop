# Feature: Runtime hardening for jobs, threads, and LLM failure handling

## Scenario: job config persistence is safe across concurrent writers
Given multiple backend processes can update `.reddrop/jobs_config.json`
When a job is created, updated, activated, or deleted
Then job config writes use file locking and atomic replace semantics
And the config file is never persisted via direct non-atomic truncating writes

## Scenario: threads listing avoids repeated artifact re-reads and full global sort work
Given the frontend refreshes threads every 2 seconds
When runs artifacts are unchanged
Then backend thread listing reuses cached thread rows instead of re-reading every artifact file
And pagination returns top-ranked threads without sorting the full matching set in memory

## Scenario: reply linkage is explicit and thread actions accept reply-job references
Given a Search job has one or more linked Reply jobs
When the backend returns rows from `GET /threads`
Then each row includes linked reply metadata (`reply_job_id`, `reply_job_name`, personas)
And per-thread reply/send endpoints can be called with either a Search job ref or a linked Reply job ref

## Scenario: LLM unavailability or request failure fails the run
Given OpenRouter is missing or unavailable
When reply generation or search analysis requests the LLM
Then the backend raises an explicit runtime failure
And it does not silently return demo/fallback generated reply text

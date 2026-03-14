# Feature: Architecture document coherence

## Scenario: documented CLI entrypoint matches repository layout
Given the architecture document lists CLI entrypoints
When repository files are inspected
Then the documented entrypoint path matches existing executable scripts

## Scenario: runtime execution model is accurately described
Given job runs are started from API runtime controls
When architecture behavior is documented
Then it describes in-process command handler execution
And it does not claim subprocess spawning behavior that is not implemented

## Scenario: settings persistence wording matches implementation
Given runtime settings are loaded/saved by settings store
When architecture documentation describes settings persistence
Then it references `settings.yaml` backed storage with environment synchronization

## Scenario: auto-refresh defaults are consistent with frontend behavior
Given Jobs and Threads pages configure auto-refresh state
When architecture documentation describes default refresh behavior
Then default value matches current UI state

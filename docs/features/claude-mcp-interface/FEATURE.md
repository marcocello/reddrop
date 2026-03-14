# Claude MCP Interface for FastAPI and CLI Workflows

## Acceptance Scenarios
Feature: Claude-driven MCP control of Reddrop workflows
  In order to let Claude automate Reddrop operations safely and predictably
  As an operator integrating AI-assisted runtime control
  I want an MCP interface that exposes approved backend capabilities with consistent behavior across FastAPI and CLI

  Scenario: Claude can discover available MCP tools and their argument contracts
    Given the backend is running with MCP enabled
    When Claude initializes an MCP session
    And Claude requests the available tools
    Then the response includes a deterministic list of tool names
    And each tool exposes a machine-readable schema for required and optional arguments
    And each tool description is sufficient to invoke the tool without out-of-band instructions

  Scenario: Claude executes a search job through MCP successfully
    Given a configured active search job named "daily_reddit"
    And the backend has valid credentials for that job social platform
    When Claude invokes the MCP tool to run search for job "daily_reddit"
    Then the tool response reports success
    And the response includes the job name and generated artifact reference
    And a subsequent thread listing includes rows belonging to "daily_reddit"

  Scenario: MCP search behavior is consistent with existing FastAPI behavior
    Given a configured active search job named "daily_reddit"
    When I run search through FastAPI for job "daily_reddit"
    And I run the equivalent search through MCP for job "daily_reddit"
    Then both responses indicate the same job identity
    And both responses expose equivalent success or failure state
    And both produce an addressable search artifact for the same job

  Scenario: MCP command behavior is consistent with existing CLI behavior
    Given a configured active search job named "daily_reddit"
    When I run `reddrop search daily_reddit` from the CLI
    And Claude runs the equivalent MCP search tool for "daily_reddit"
    Then both executions complete without backend crashes
    And both report machine-readable success metadata
    And both identify output for the same job name

  Scenario: MCP rejects unknown tools with actionable error information
    Given the backend is running with MCP enabled
    When Claude invokes a non-existent MCP tool named "jobs.destroy_all"
    Then the request is rejected with a structured error
    And the error code indicates unknown tool
    And the error payload includes available valid tool names

  Scenario Outline: MCP rejects invalid input before backend mutation
    Given the backend is running with MCP enabled
    When Claude invokes tool "<tool>" with invalid arguments
    Then the response has validation status "error"
    And the response clearly identifies the invalid field "<field>"
    And no jobs, artifacts, or auth records are changed

    Examples:
      | tool         | field       |
      | jobs.search  | name        |
      | threads.reply| persona     |
      | threads.send | thread_id   |

  Scenario: MCP enforces authorization for tool execution
    Given the backend is running with MCP enabled and access control configured
    When Claude invokes an MCP tool without valid authorization context
    Then tool execution is denied
    And the response status is unauthorized
    And no CLI-equivalent operation is executed

  Scenario: MCP request timeout is surfaced as deterministic failure
    Given the backend is running with MCP enabled
    And the MCP execution timeout is configured to 5 seconds
    When Claude invokes a tool call that exceeds the timeout
    Then the response status is timeout
    And the response includes the timed-out tool name
    And the backend remains healthy for subsequent requests

  Scenario Outline: MCP transport mode preserves tool contract
    Given the backend is started with MCP transport "<transport>"
    When Claude performs MCP initialize and tool listing
    Then the initialize response is valid
    And the same tool names are available regardless of transport mode

    Examples:
      | transport |
      | stdio     |
      | http      |

## Constraints
- MCP tool calls must expose only allowlisted operations that map to existing Reddrop workflows.
- MCP results must be structured JSON with stable top-level fields: `status`, `tool`, and `data` or `error`.
- Input validation must happen before any mutation and must return field-level errors.
- Behavior exposed via MCP must remain observably consistent with existing FastAPI and CLI outcomes for equivalent operations.
- The feature must avoid adding hidden side effects outside explicitly invoked tool actions.

## Definition of Done
- Acceptance scenarios in this document are executable through feature-scoped acceptance tests.
- The backend can be controlled by Claude via MCP for approved operations without bypassing existing validation rules.
- Equivalent operations across MCP, FastAPI, and CLI produce consistent observable outcomes.
- Repo gate and feature acceptance checks pass for this feature directory.

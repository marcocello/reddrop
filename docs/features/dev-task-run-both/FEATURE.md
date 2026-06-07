# Dev Task Run Both

## Behavior

Developers need one VS Code task that starts the Python backend and Vite
frontend together for local development.

## Scenarios

```gherkin
Feature: Dev task run both

  Scenario: Start backend and frontend from one task
    Given the repository has a VS Code task configuration
    When a developer runs the "Run Backend and Frontend" task
    Then the backend server starts from the repository root on port 8000
    And the frontend dev server starts from the frontend directory
    And both child tasks run in parallel
```

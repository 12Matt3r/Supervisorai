# AI Supervisor & Orchestrator

## 1. Project Overview

This project is a sophisticated, AI-powered system designed to supervise, manage, and assist other AI agents. It has evolved from a simple monitoring script into a multi-layered platform with advanced capabilities for intelligent oversight and autonomous operation.

The system is built around three core concepts:
*   **Supervision:** A supervisor agent that uses a probabilistic model (Expectimax) to watch over a working agent, predict potential issues, and intervene when necessary.
*   **Orchestration:** An autonomous orchestrator that can manage a pool of specialized agents, decompose high-level goals into a dependency graph of tasks, and manage the entire execution workflow.
*   **Assistance:** A proactive research assistant that can detect when an agent is stuck, perform web searches to find solutions for its errors, and provide intelligent suggestions to help it recover.

## 2. Core Features

This project includes a rich set of features built progressively:

*   **Intelligent Supervisor Agent:**
    *   Uses an **Expectimax algorithm** to make nuanced decisions about whether to `ALLOW`, `WARN`, `CORRECT`, or `ESCALATE` an agent's output.
    *   The decision-making is based on a weighted evaluation of the agent's state, including output quality, task drift, error count, and resource usage.

*   **Feedback-Driven Learning:**
    *   The supervisor can **learn from user feedback**. The dashboard allows a human to correct a bad decision, and this feedback is used to retrain the weights of the Expectimax agent's evaluation function.
    *   This creates a powerful self-improvement loop, allowing the supervisor's judgment to get better over time.

*   **Interactive Debugger & Dashboard:**
    *   A comprehensive web dashboard (`examples/dashboard.html`) serves as the central UI.
    *   It features an **interactive debugger** that visualizes the Expectimax agent's entire decision tree as a flowchart, allowing for deep "what-if" analysis.
    *   It also includes panels for idea validation, decision logging, and orchestrator status.

*   **Autonomous Orchestrator:**
    *   Manages a pool of specialized agents with different capabilities.
    *   Features an **LLM-powered task planner** that can take a high-level goal (e.g., "build a web scraper") and autonomously decompose it into a detailed, multi-step plan with dependencies.
    *   The orchestrator executes this plan by assigning tasks to available agents and uses the `SupervisorCore` to monitor each step.

*   **Proactive Research Assistant:**
    *   The supervisor can detect when an agent is "stuck" (e.g., failing repeatedly).
    *   It then autonomously formulates a search query, uses **Google Search** to find relevant help articles (e.g., on Stack Overflow), reads the content, and uses an **LLM to synthesize a helpful suggestion**.
    *   This suggestion is delivered as a new `ASSISTANCE` intervention, allowing the system to solve its own problems without human intervention.

## 3. System Architecture

The project is organized into a standard Python project structure:

*   `src/supervisor_agent/`: Contains the core `SupervisorCore` and the `ExpectimaxAgent`.
*   `src/orchestrator/`: Contains the `Orchestrator` and its data models for managing projects and tasks.
*   `src/researcher/`: Contains the `ResearchAssistor` responsible for proactive help.
*   `src/llm/`: Contains the generic `LLMClient` for interacting with language models.
*   `src/server/`: Contains the `FastMCP` server (`main.py`) that exposes all functionality through a WebSocket-based API.
*   `examples/dashboard.html`: The all-in-one web interface for interacting with the system.
*   `tests/`: Contains unit and integration tests for the various components.

## 4. Setup and Installation

To get the project running, follow these steps:

1.  **Set up a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    The project's dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys (Optional):**
    The system uses the Anthropic Claude 3 API for its LLM capabilities. To enable this, set the following environment variable:
    ```bash
    export ANTHROPIC_API_KEY="your-api-key-here"
    ```
    If the API key is not set, the LLM-powered features will return mocked responses, but the rest of the system will still be functional.

## 5. How to Run the System

1.  **Start the Server:**
    Run the main server file from the root of the project:
    ```bash
    python3 src/server/main.py
    ```
    You should see output indicating that the MCP server, supervisor, and orchestrator have started.

2.  **Use the Dashboard:**
    Open the `examples/dashboard.html` file in your web browser. This file is self-contained and will connect to the local server automatically.

## 6. How to Use the Dashboard

The dashboard provides a comprehensive interface for all the system's features:

*   **Idea Validation:** Enter an idea and get a validation report.
*   **Manual Decision Test:** Manually adjust the features of an agent's state and get a single decision from the supervisor.
*   **Decision Dashboard:** View a chart of decision scores over time and a raw log of all supervisor interventions. You can provide feedback on decisions here to help the supervisor learn.
*   **Interactive Debugger:** Set up a "what-if" scenario and generate a full, interactive flowchart of the supervisor's entire decision-making process. Click on nodes in the graph to inspect the agent's state at that point.
*   **Autonomous Orchestrator:**
    1.  Click **"Register Sample Agent"** to add a capable agent to the pool.
    2.  Enter a high-level goal in the "Submit New Goal" section (e.g., "Create a Python script to scrape a website").
    3.  Click **"Submit Goal"**. The orchestrator will use an LLM to generate a task plan, which will be displayed as a flowchart.
    4.  The system will then begin executing the plan, and you can watch the status of the tasks update in real-time.

# AI Agent Supervisor

This project contains a comprehensive supervisor agent designed to oversee and manage other AI agents. It features a proactive MiniMax agent for making intervention decisions, a full suite of monitoring and reporting tools, and a browser extension for real-time interaction.

This repository has been refactored into a clean, standard project structure.

## Project Structure

The project is now organized into the following directories:

-   `src/`: Contains all the Python source code for the supervisor agent and its components.
    -   `src/supervisor_agent/`: The core Python package for the supervisor, including the `core.py` logic, the `__init__.py` with data class definitions, and the new `minimax_agent.py`.
    -   `src/monitoring/`, `src/reporting/`, `src/error_handling/`: Sub-packages for the different components of the supervisor.
    -   `src/server/`: Contains the `FastMCP` server code.
-   `tests/`: Contains all the test files.
-   `browser_extension/`: Contains the files for the browser extension frontend.
-   `docs/`: Contains all the original documentation files.
-   `examples/`: Contains various demo scripts and the `minimax.html` interface.
-   `scripts/`: Contains utility and startup scripts like `run.sh`.

## Setup and Installation

This project uses `uv` for package and environment management.

1.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install all dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file may need to be generated from `pyproject.toml`. For now, you can install the key dependencies manually: `uv pip install "fastmcp>=0.5.0" pydantic psutil jinja2 pandas matplotlib seaborn aiofiles`)*

## How to Run

### Running the Supervisor Server

The main application is the supervisor server. You can run it using the provided script, which will start the `FastMCP` server on `ws://localhost:8765`.

1.  **Set the Python Path:**
    You need to add the `src` directory to your `PYTHONPATH` so that the server can find the Python packages.
    ```bash
    export PYTHONPATH=$(pwd)/src
    ```

2.  **Run the server:**
    ```bash
    python src/server/main.py
    ```
    *(Note: The `scripts/run.sh` is configured to run `src/server/server.py`, which is a duplicate. It is recommended to use `src/server/main.py` directly).*

### Using the MiniMax Agent Interface

The `minimax.html` file provides a simple web interface to test the decision-making of the MiniMax agent.

1.  Make sure the supervisor server is running (see above).
2.  Open the `examples/minimax.html` file in your web browser.
3.  Adjust the sliders and input values to simulate different agent states.
4.  Click the "Get Decision" button to see the supervisor's recommended action.

## Testing

The project contains two main test suites:

-   **`tests/test_minimax_agent.py`**: This is a suite of unit tests for the new MiniMax agent. It is fully functional and passes. You can run it with:
    ```bash
    python tests/test_minimax_agent.py
    ```

-   **`tests/test_comprehensive.py`**: This is the original, comprehensive test suite for the project. **Note: This test suite is currently broken.** It contains numerous errors related to missing dependencies and incorrect assumptions about the code. While some issues have been fixed, fully repairing this test suite is a larger task that is outside the scope of the recent work.

# Instructions for AI Agents

This document provides explicit, step-by-step instructions for an AI agent to set up and run this project. Assume you have access to a shell environment and the ability to execute the commands listed here. Do not deviate from these instructions.

## Objective

Your objective is to successfully run the main application server, which provides a WebSocket API for supervising and orchestrating other AI agents.

## Step-by-Step Instructions

Follow these steps in order without skipping any.

### Step 1: Verify Your Current Directory

Your first step is to verify that you are in the root directory of the project. You can do this by listing the files and ensuring you see the `src`, `tests`, and `requirements.txt` files.

**Command:**
```bash
ls
```

**Expected Output (should include these files/directories):**
```
AGENTS.md
README.md
config/
docs/
examples/
requirements.txt
scripts/
src/
tests/
```
If you do not see these files, you are in the wrong directory. Navigate to the correct project root before proceeding.

### Step 2: Create a Python Virtual Environment

You must create and activate a Python virtual environment to isolate the project's dependencies. Do not skip this step. The required tool `uv` is assumed to be installed.

**Command:**
```bash
uv venv
```

**Expected Output:**
You should see output indicating the creation of a virtual environment in a `.venv` directory.

### Step 3: Activate the Virtual Environment

You must activate the virtual environment you just created.

**Command:**
```bash
source .venv/bin/activate
```

**Expected Output:**
Your shell prompt should now be prefixed with `(.venv)`, indicating that you are inside the virtual environment.

### Step 4: Install All Required Dependencies

Install all required Python packages using the `requirements.txt` file. Use the exact command below.

**Command:**
```bash
pip install -r requirements.txt
```

**Expected Output:**
You will see a series of lines indicating that packages are being downloaded and installed. Wait for this command to complete successfully. There should be no errors.

### Step 5: Run the Main Application Server

Now, you will start the main server. This is the final goal. The server provides a `FastMCP` WebSocket API.

**Command:**
```bash
python3 src/server/main.py
```

**Expected Output:**
The server will start and log several lines to the console. You should see messages indicating that the Supervisor, Orchestrator, and MCP server have been initialized. The command will not exit, as it is a running server. It will look something like this:

```
2025-08-21 XX:XX:XX - __main__ - INFO - Starting Integrated Supervisor MCP Server
2025-08-21 XX:XX:XX - orchestrator.core - INFO - Orchestrator started.
2025-08-21 XX:XX:XX - supervisor_agent.core - INFO - Supervisor initialized: SupervisorCore
2025-08-21 XX:XX:XX - __main__ - INFO - Integration mode: Basic
2025-08-21 XX:XX:XX - fastmcp - INFO - Starting FastMCP server on ws://localhost:8765
```

Once you see this output, you have successfully completed your objective. The server is now running. Do not proceed to any other steps unless given a new objective.

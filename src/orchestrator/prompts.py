from typing import List, Dict, Any

def get_decomposition_prompt(goal: str, agents: List[Dict[str, Any]]) -> str:
    """
    Generates a prompt for the LLM to decompose a high-level goal into a task graph.

    Args:
        goal: The user's high-level goal.
        agents: A list of available agents and their capabilities.

    Returns:
        A string containing the formatted prompt.
    """

    agent_capabilities_str = "\n".join(
        f"- Agent '{agent['name']}' (ID: {agent['agent_id']}) can perform: {', '.join(agent['capabilities'])}"
        for agent in agents
    )

    prompt = f"""
You are an expert project manager AI. Your task is to take a high-level user goal and decompose it into a structured plan of tasks that can be executed by a team of specialized AI agents.

**Your Goal:**
Decompose the following user request into a series of tasks with dependencies.

**User Request:**
"{goal}"

**Available Agents and Their Capabilities:**
{agent_capabilities_str}

**Instructions:**
1.  Analyze the user's request and break it down into a logical sequence of smaller, concrete tasks.
2.  For each task, provide a unique `task_id`, a short `name`, a clear `description`, and a list of the `required_capabilities` from the available agents.
3.  Define the `dependencies` for each task. A task's dependencies should be a list of `task_id`s that must be completed before this task can start. The first task(s) should have an empty dependency list.
4.  Ensure the plan is logical and covers all aspects of the user's request.
5.  You MUST return your response as a single, valid JSON object. The JSON object should represent the project plan.

**JSON Output Format:**
The JSON object must have a single key, "tasks", which is an object where each key is a `task_id` and the value is the task details object.

{{
  "tasks": {{
    "task_id_1": {{
      "name": "Task Name 1",
      "description": "A clear description of what needs to be done for this task.",
      "required_capabilities": ["capability1", "capability2"],
      "dependencies": []
    }},
    "task_id_2": {{
      "name": "Task Name 2",
      "description": "A description for the second task.",
      "required_capabilities": ["capability3"],
      "dependencies": ["task_id_1"]
    }}
  }}
}}

**Example:**
If the user request is "Create a simple website and deploy it" and the agents have capabilities for "frontend_dev", "backend_dev", and "deployment", your output might look like this:

{{
  "tasks": {{
    "design_frontend": {{
      "name": "Design Frontend",
      "description": "Create the HTML, CSS, and JavaScript for the website's user interface.",
      "required_capabilities": ["frontend_dev"],
      "dependencies": []
    }},
    "develop_backend": {{
      "name": "Develop Backend API",
      "description": "Create a simple backend API to serve data to the frontend.",
      "required_capabilities": ["backend_dev"],
      "dependencies": []
    }},
    "integrate_frontend_backend": {{
      "name": "Integrate Frontend and Backend",
      "description": "Connect the frontend to the backend API to ensure they work together.",
      "required_capabilities": ["frontend_dev", "backend_dev"],
      "dependencies": ["design_frontend", "develop_backend"]
    }},
    "deploy_website": {{
      "name": "Deploy Website",
      "description": "Deploy the integrated website to a cloud hosting provider.",
      "required_capabilities": ["deployment"],
      "dependencies": ["integrate_frontend_backend"]
    }}
  }}
}}

Now, please generate the JSON task plan for the user request provided above.
"""
    return prompt

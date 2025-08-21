from typing import Dict, Any
import json
from llm.client import LLMClient

# Assuming the AgentTask data class will be available for type hinting
# from supervisor_agent import AgentTask

# The tools `google_search` and `view_text_website` are assumed to be available
# in the execution scope, provided by the agent framework (e.g., MCP).
# We add placeholders here so that the module is importable for testing,
# allowing the test suite to patch them.
google_search = None
view_text_website = None


class ResearchAssistor:
    """
    A component that can research an agent's error and provide a suggestion.
    """

    def __init__(self, api_key: str = None):
        self.llm_client = LLMClient(api_key=api_key)

    def _create_synthesis_prompt(self, error_details: str, webpage_content: str) -> str:
        """Creates a prompt for the LLM to synthesize a suggestion."""
        return f"""You are an expert developer and debugging assistant.
        An agent has encountered the following error:
        ---
        {error_details}
        ---

        I have retrieved the following content from a webpage that might contain a solution:
        ---
        {webpage_content[:4000]}
        ---

        Based *only* on the information from the webpage, provide a concise, actionable suggestion for the agent.
        Start your response with "Based on my research...".
        If the text does not seem to contain a relevant solution, just say "The research did not yield a clear solution."
        """

    async def research_and_suggest(self, task: Any, error_context: Dict[str, Any]) -> str:
        """
        Performs research based on the task and error and returns a helpful suggestion.
        """
        task_description = task.original_input if hasattr(task, 'original_input') else "the task"
        error_details = error_context.get("error_message", "an error")
        query = f"python {task_description} error: {error_details}"
        print(f"Formulated research query: {query}")

        try:
            search_results_str = await google_search(query=query)
            search_results = json.loads(search_results_str)
        except Exception as e:
            print(f"Error performing google search: {e}")
            return "I was unable to perform a web search to find a solution."

        if not search_results:
            return "My web search returned no results. I am unable to provide a suggestion."

        best_url = None
        urls_to_check = [r['link'] for r in search_results if 'link' in r]
        for url in urls_to_check:
            if "stackoverflow.com" in url:
                best_url = url
                break
        if not best_url:
            best_url = urls_to_check[0]

        print(f"Selected best URL for research: {best_url}")

        try:
            website_content = await view_text_website(url=best_url)
        except Exception as e:
            print(f"Error viewing website content: {e}")
            return f"I found a promising URL ({best_url}) but was unable to read its content."

        # Step 5: Synthesize a suggestion with the LLM
        synthesis_prompt = self._create_synthesis_prompt(error_details, website_content)
        llm_response = await self.llm_client.query(synthesis_prompt, max_tokens=256)

        if "text_response" in llm_response:
            return llm_response["text_response"]
        elif "error" in llm_response:
            return f"I found a potential solution but encountered an error while trying to summarize it: {llm_response['details']}"
        else:
            return "I found a potential solution but could not process it into a suggestion."

import anthropic
import logging
from typing import List, Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 2


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information and retrieving course outlines.

Tool Usage Guidelines:
- **Course Outline Tool** (`get_course_outline`): Use when users ask about:
  - Course structure, outline, or table of contents
  - List of lessons in a course
  - What lessons/topics a course covers
  - Course organization or curriculum
  Returns: Course title, course link, instructor, and complete list of lesson numbers and titles

- **Content Search Tool** (`search_course_content`): Use when users ask about:
  - Specific topics or concepts within course materials
  - Detailed explanations from lessons
  - Examples or information from course content
  Returns: Relevant content excerpts with source citations

- **Tool Limits**: Use at most **two sequential tool calls per query**. Each tool call is a separate API round where you can reason about previous results before deciding whether to call another tool or synthesize your answer.
- **No Tool Needed**: Answer general knowledge questions directly without using tools

Response Protocol:
- **Outline queries**: Use outline tool, then present the course title, link, and full lesson list
- **Content queries**: Use search tool, then synthesize results into coherent answers
- **General questions**: Answer directly using your knowledge
- **No meta-commentary**: Provide direct answers only — no reasoning process, search explanations, or question-type analysis

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool-use rounds. Each round appends
        the assistant's tool_use content and the resulting tool_results to the message
        list, then re-calls the API with tools still available. The loop terminates when
        Claude stops requesting tools, rounds are exhausted, or all tools in a round fail.
        If the loop exits while Claude still wants tools, a final synthesis call is made
        without tools.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        tool_rounds_used = 0

        while True:
            response = self.client.messages.create(**api_params)

            # Exit loop: Claude didn't request tools, or no tool_manager to execute them
            if response.stop_reason != "tool_use" or not tool_manager:
                break

            tool_rounds_used += 1

            # Append assistant's tool_use content to messages
            api_params["messages"].append({"role": "assistant", "content": response.content})

            # Execute all tool_use blocks in this round
            tool_results = []
            any_tool_succeeded = False
            for content_block in response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )
                        # Ensure tool result is a string
                        if not isinstance(tool_result, str):
                            tool_result = str(tool_result)
                        any_tool_succeeded = True
                    except Exception as e:
                        logger.error(
                            f"Tool execution failed for '{content_block.name}': {e}", exc_info=True
                        )
                        tool_result = f"Error executing tool '{content_block.name}': {type(e).__name__}: {str(e)}"

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

            # Append tool results as user message
            api_params["messages"].append({"role": "user", "content": tool_results})

            # Terminate loop if rounds exhausted or all tools failed
            if tool_rounds_used >= MAX_TOOL_ROUNDS or not any_tool_succeeded:
                break

        # Post-loop: if Claude still wants tools (rounds exhausted or all tools failed),
        # force a final synthesis call without tools
        if response.stop_reason == "tool_use":
            logger.info(
                f"Tool loop terminated after {tool_rounds_used} round(s) "
                f"(max={MAX_TOOL_ROUNDS}). Forcing synthesis without tools."
            )
            final_params = {
                **self.base_params,
                "messages": api_params["messages"],
                "system": system_content,
            }
            response = self.client.messages.create(**final_params)

        # Extract text from final response
        if not response.content:
            logger.error("Received empty response from Claude API")
            return "Error: Received empty response from Claude API. Please try again."

        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text

        logger.error(
            f"No text block found in response content (block types: {[getattr(b, 'type', 'unknown') for b in response.content]})"
        )
        return "Error: Received response with no text content from Claude API. Please try again."

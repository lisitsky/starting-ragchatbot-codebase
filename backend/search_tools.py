from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from vector_store import VectorStore, SearchResults

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        try:
            results = self.store.search(
                query=query,
                course_name=course_name,
                lesson_number=lesson_number
            )
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return f"Search error: {type(e).__name__}: {str(e)}"

        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track structured sources for the UI

        # Batch lookup: collect unique course-lesson combinations
        unique_sources = {}
        for meta in results.metadata:
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            key = (course_title, lesson_num)
            if key not in unique_sources:
                unique_sources[key] = None

        # Lookup all links at once (prevents N+1 queries)
        for (course_title, lesson_num) in unique_sources.keys():
            if lesson_num is not None:
                # Try lesson link first
                url = self.store.get_lesson_link(course_title, lesson_num)
                if not url:
                    # Fallback to course link
                    url = self.store.get_course_link(course_title)
            else:
                # No lesson number - use course link
                url = self.store.get_course_link(course_title)
            unique_sources[(course_title, lesson_num)] = url

        # Process results with link information
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build context header for search results
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Build source citation text
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"

            # Get URL from batch lookup
            url = unique_sources.get((course_title, lesson_num))

            # Create structured source (dict format)
            source_citation = {
                "text": source_text,
                "url": url
            }
            sources.append(source_citation)

            formatted.append(f"{header}\n{doc}")

        # Store structured sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving complete course outlines with lesson lists"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last query

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the complete outline/structure of a course including all lesson numbers and titles",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title or partial name (e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the course outline retrieval.

        Args:
            course_name: Course name or partial match

        Returns:
            Formatted course outline with all lessons
        """
        import json

        # Resolve the course name to exact title
        course_title = self.store._resolve_course_name(course_name)

        if not course_title:
            return f"No course found matching '{course_name}'"

        # Get the course metadata
        try:
            results = self.store.course_catalog.get(ids=[course_title])

            if not results or not results.get('metadatas'):
                return f"Course '{course_name}' not found in catalog"

            metadata = results['metadatas'][0]

            # Extract course information
            title = metadata.get('title', 'Unknown')
            course_link = metadata.get('course_link')
            instructor = metadata.get('instructor', 'Unknown')
            lessons_json = metadata.get('lessons_json', '[]')

            # Parse lessons
            lessons = json.loads(lessons_json)

            # Build outline response
            outline_parts = [
                f"Course: {title}",
                f"Instructor: {instructor}"
            ]

            if course_link:
                outline_parts.append(f"Course Link: {course_link}")

            outline_parts.append(f"\nLessons ({len(lessons)} total):")

            # Add each lesson
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number')
                lesson_title = lesson.get('lesson_title', 'Untitled')
                outline_parts.append(f"  Lesson {lesson_num}: {lesson_title}")

            # Track source for UI
            if course_link:
                self.last_sources = [{
                    "text": title,
                    "url": course_link
                }]
            else:
                self.last_sources = []

            return "\n".join(outline_parts)

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
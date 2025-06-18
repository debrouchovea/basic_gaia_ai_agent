from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from typing import ClassVar, Optional

class TextInspectorInput(BaseModel):
    file_path: str = Field(
        description="Path to the file to read (e.g., .pdf, .docx). Not for images/HTML - use other tools instead!"
    )
    question: Optional[str] = Field(
        default=None,
        description="Optional question about the file content. Omit to get raw content."
    )

class TextInspectorTool(BaseTool):
    name: str = "inspect_file_as_text"
    description: str = """
Reads files as markdown text. If a question is asked to the text, then a response will be returned. If there is no question asked, then the file content will be fully retranscribed. 
Handles: 
[".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"] 
and other text files. DOES NOT HANDLE IMAGES."""
    args_schema: type[BaseModel] = TextInspectorInput
    output_type: ClassVar[str] = "content"
    
    _model: object = PrivateAttr()
    _text_limit: int = PrivateAttr()
    _md_converter: object = PrivateAttr()

    def __init__(self, model: object = None, text_limit: int = 100000, **data):
        super().__init__(**data)
        # print('initializing TextInspectorTool')
        self._model = model
        self._text_limit = text_limit
        
        # Assuming MarkdownConverter is available in your environment
        from tools.mdconvert import MarkdownConverter  
        self._md_converter = MarkdownConverter()

    def _run(self, file_path: str, question: Optional[str] = None) -> str:

        # Handle image files
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("Use visualizer tool for images, not this text tool!")

        # Convert file to markdown
        result = self._md_converter.convert(file_path)

        # Handle ZIP files
        if ".zip" in file_path:
            return result.text_content

        # Return raw content if no question
        if not question:
            return result.text_content

        # Build message chain based on content size
        if len(result.text_content) < 100000000000000000000:
            return f"Document content: {result.text_content}"

        # Construct AI messages

        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Answer this question with three sections:\n{question}\n"
                            "Use headings: 1. Short answer, 2. Detailed answer, 3. Context"},
                        {
                    "type": "text",
                    "text": f"File: {result.title}\n\n{result.text_content[:self._text_limit]}"
                        }
                    ]
                }
            ]

        output = self._model.invoke(messages)
        return output.content
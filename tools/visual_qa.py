from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, ClassVar, Any
import base64
import mimetypes
import os
import uuid
import requests
from io import BytesIO
import PIL.Image

class VisualQAInput(BaseModel):
    image_path: str = Field(description="Local path or URL of the image to analyze")
    question: Optional[str] = Field(
        default=None, 
        description="Question about the image. If omitted, generates a caption"
    )

class VisualQATool(BaseTool):
    name: str = "visualizer"
    description: str = "Analyzes images and answers questions about their content"
    args_schema: ClassVar[type[BaseModel]] = VisualQAInput
    output_type: ClassVar[str] = "content"
    
    _llm: Any = PrivateAttr()
    _model_provider: str = PrivateAttr()

    def __init__(self, llm: Any = None, model_provider: str = "openai", **kwargs):
        super().__init__(**kwargs)
        if llm is None:
            # Default to your init_chat_model pattern
            from langchain.chat_models import init_chat_model
            self._llm = init_chat_model("gpt-4.1", model_provider=model_provider, temperature=0)
        else:
            self._llm = llm
        self._model_provider = model_provider

    def _encode_image(self, image_path: str) -> str:
        """Handle both local and remote images"""
        if image_path.startswith("http"):
            image_path = self._download_remote_image(image_path)
            
        with open(image_path, "rb") as image_file:
            print('image_path in _encode_image:', image_path)
            # import os
            # print('filezise', os.path.getsize(image_path))
            # from pathlib import Path
            # p = Path(image_path)
            # print('infooo', p.exists(), p.is_file(), p.stat().st_size)
            
            # raw = image_file.read()
            # print("Bytes read:", len(raw))

            base =  base64.b64encode(image_file.read()).decode("utf-8")
            print('base in _encode_image:', base)
            return base
    def _download_remote_image(self, url: str) -> str:
        """Download remote image to local temp file"""
        response = requests.get(url, headers={"User-Agent": "Langgraph/1.0"})
        response.raise_for_status()
        
        ext = mimetypes.guess_extension(response.headers.get("content-type", "")) or ".bin"
        path = os.path.join("downloads", f"{uuid.uuid4()}{ext}")
        
        with open(path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
                
        return path

    def _run(self, image_path: str, question: Optional[str] = None) -> str:
        add_note = False
        if not question:
            question = "Please describe this image in detail"
            add_note = True

        mime_type, _ = mimetypes.guess_type(image_path)
        base64_image = self._encode_image(image_path)

        # Construct LLM input based on provider
        if self._model_provider == "openai":
            content = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question},
                                    {
                                        "type": "image_url" if self._model_provider == "openai" else "image",
                                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                                    }
                                ]
                            }
                        ]

        else:  # For other providers using your init pattern
            content = [
                {"type": "image", "data": base64_image, "mime_type": mime_type},
                {"type": "text", "text": question},
            ]
        print('content in visual qa tool:', content)
        # Use the injected LLM
        response = self._llm.invoke(content)
        output = response.content if hasattr(response, "content") else str(response)

        if add_note:
            output = f"Generated description: {output}"
        return output

# from langchain_core.tools import BaseTool
# from pydantic import BaseModel, Field, PrivateAttr
# from typing import Optional, ClassVar
# import base64
# import mimetypes
# import os
# import uuid
# import requests
# from io import BytesIO
# import PIL.Image

# class VisualQAInput(BaseModel):
#     image_path: str = Field(description="Local path or URL of the image to analyze")
#     question: Optional[str] = Field(
#         default=None, 
#         description="Question about the image. If omitted, generates a caption"
#     )

# class VisualQATool(BaseTool):
#     name: str = "visualizer"
#     description: str = "Analyzes images and answers questions about their content"
#     args_schema: ClassVar[type[BaseModel]] = VisualQAInput
#     output_type: ClassVar[str] = "content"
    
#     _api_key: str = PrivateAttr()

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._api_key = os.getenv("OPENAI_API_KEY")

#     def _encode_image(self, image_path: str) -> str:
#         """Handle both local and remote images"""
#         if image_path.startswith("http"):
#             image_path = self._download_remote_image(image_path)
            
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode("utf-8")

#     def _download_remote_image(self, url: str) -> str:
#         """Download remote image to local temp file"""
#         response = requests.get(url, headers={"User-Agent": "Langgraph/1.0"})
#         response.raise_for_status()
        
#         ext = mimetypes.guess_extension(response.headers.get("content-type", "")) or ".bin"
#         path = os.path.join("downloads", f"{uuid.uuid4()}{ext}")
        
#         with open(path, "wb") as f:
#             for chunk in response.iter_content(1024):
#                 f.write(chunk)
                
#         return path

#     def _run(self, image_path: str, question: Optional[str] = None) -> str:
#         # Set default question if not provided
#         add_note = False
#         if not question:
#             question = "Please describe this image in detail"
#             add_note = True

#         # Encode image
#         mime_type, _ = mimetypes.guess_type(image_path)
#         base64_image = self._encode_image(image_path)

#         # Build OpenAI request
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self._api_key}"
#         }
        
#         payload = {
#             "model": "gpt-4o",
#             "messages": [{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": question},
#                     {"type": "image_url", "image_url": {
#                         "url": f"data:{mime_type};base64,{base64_image}"
#                     }}
#                 ]
#             }],
#             "max_tokens": 10000
#         }

#         # Send request
#         response = requests.post(
#             "https://api.openai.com/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )
        
#         try:
#             output = response.json()["choices"][0]["message"]["content"]
#         except KeyError:
#             raise ValueError(f"Unexpected API response: {response.text}")

#         if add_note:
#             output = f"Generated description: {output}"

#         return output
    


# from langchain_core.tools import BaseTool
# from pydantic import BaseModel, Field, PrivateAttr
# from typing import Optional, ClassVar, Any
# import base64
# import mimetypes
# import os
# import uuid
# import requests
# from io import BytesIO
# import PIL.Image

# class VisualQAInput(BaseModel):
#     image_path: str = Field(description="Local path or URL of the image to analyze")
#     question: Optional[str] = Field(
#         default=None, 
#         description="Question about the image. If omitted, generates a caption"
#     )

# class VisualQATool(BaseTool):
#     name: str = "visualizer"
#     description: str = "Analyzes images and answers questions about their content"
#     args_schema: ClassVar[type[BaseModel]] = VisualQAInput
#     output_type: ClassVar[str] = "content"
    
#     _model: Any = PrivateAttr()

#     def __init__(self, model: Any = None, **kwargs):
#         super().__init__(**kwargs)
#         self._model = model or self._default_openai_model

#     @property
#     def _default_openai_model(self):
#         """Fallback to OpenAI if no model provided"""
#         class OpenAIWrapper:
#             def __call__(self, image_data: str, mime_type: str, question: str) -> str:
#                 headers = {
#                     "Content-Type": "application/json",
#                     "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
#                 }
                
#                 payload = {
#                     "model": "gpt-4o",
#                     "messages": [{
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": question},
#                             {"type": "image_url", "image_url": {
#                                 "url": f"data:{mime_type};base64,{image_data}"
#                             }}
#                         ]
#                     }],
#                     "max_tokens": 10000
#                 }

#                 response = requests.post(
#                     "https://api.openai.com/v1/chat/completions",
#                     headers=headers,
#                     json=payload
#                 )
                
#                 try:
#                     return response.json()["choices"][0]["message"]["content"]
#                 except KeyError:
#                     raise ValueError(f"Unexpected API response: {response.text}")

#         return OpenAIWrapper()

#     def _encode_image(self, image_path: str) -> str:
#         """Handle both local and remote images"""
#         if image_path.startswith("http"):
#             image_path = self._download_remote_image(image_path)
            
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode("utf-8")

#     def _download_remote_image(self, url: str) -> str:
#         """Download remote image to local temp file"""
#         response = requests.get(url, headers={"User-Agent": "Langgraph/1.0"})
#         response.raise_for_status()
        
#         ext = mimetypes.guess_extension(response.headers.get("content-type", "")) or ".bin"
#         path = os.path.join("downloads", f"{uuid.uuid4()}{ext}")
        
#         with open(path, "wb") as f:
#             for chunk in response.iter_content(1024):
#                 f.write(chunk)
                
#         return path

#     def _run(self, image_path: str, question: Optional[str] = None) -> str:
#         add_note = False
#         if not question:
#             question = "Please describe this image in detail"
#             add_note = True

#         mime_type, _ = mimetypes.guess_type(image_path)
#         base64_image = self._encode_image(image_path)

#         # Use the injected model
#         output = self._model(
#             image_data=base64_image,
#             mime_type=mime_type,
#             question=question
#         )

#         if add_note:
#             output = f"Generated description: {output}"

#         return output
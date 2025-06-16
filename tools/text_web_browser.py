# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py

# GOT FROM HUGGING FACE: https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/scripts/text_web_browser.py
import mimetypes
import os
import pathlib
import re
import time
import uuid
from typing import Any
from urllib.parse import unquote, urljoin, urlparse
from typing import ClassVar

# import pathvalidate
import requests
from serpapi import GoogleSearch

from smolagents import Tool

from tools.cookies import COOKIES
from tools.mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, PrivateAttr


class SimpleTextBrowser:
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: str | None = None,
        viewport_size: int | None = 1024 * 8,
        downloads_folder: str | None | None = None,
        serpapi_key: str | None | None = None,
        request_kwargs: dict[str, Any] | None | None = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: list[tuple[str, float]] = list()
        self.page_title: str | None = None
        self.viewport_current_page = 0
        self.viewport_pages: list[tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.serpapi_key = serpapi_key
        self.request_kwargs = request_kwargs
        self.request_kwargs["cookies"] = COOKIES
        self._mdconvert = MarkdownConverter()
        self._page_content: str = ""

        self._find_on_page_query: str | None = None
        self._find_on_page_last_result: int | None = None  # Location of the last result

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: int | None = None) -> None:
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("google:"):
            self._serpapi_search(uri_or_path[len("google:") :].strip(), filter_year=filter_year)
        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> str | None:
        """Searches for the query from the current viewport forward, looping back to the start if necessary."""

        # Did we get here via a previous find_on_page search with the same query?
        # If so, map to find_next
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # Ok it's a new search start from the current viewport
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> str | None:
        """Scroll to the next viewport that matches the query"""

        if self._find_on_page_query is None:
            return None

        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> int | None:
        """Search for matches between the starting viewport looping when reaching the end."""

        if query is None:
            return None

        # Normalize the query, and convert to a regular expression
        nquery = re.sub(r"\*", "__STAR__", query)
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # Merge isolated stars with prior word
        nquery = nquery.replace("__STAR__", ".*").lower()

        if nquery.strip() == "":
            return None

        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0] : bounds[1]]

            # TODO: Remove markdown links and images
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def visit_page(self, path_or_uri: str, filter_year: int | None = None) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.viewport

    def _split_pages(self) -> None:
        # Do not split search results
        if self.address.startswith("google:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _serpapi_search(self, query: str, filter_year: int | None = None) -> None:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        search = GoogleSearch(params)
        results = search.get_dict()
        self.page_title = f"{query} - Search"
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(
                f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
            )
            return

        def _prev_visit(url):
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: list[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)

    def _fetch_page(self, url: str) -> None:
        download_path = ""
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # Prepare the request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                # Send a HTTP request to the URL
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                # If the HTTP request was successful
                content_type = response.headers.get("content-type", "")

                # print('======================================================================')
                # print('======================================================================')
                # print('======================================================================')
                # print('FETCH PAGE')
                # print('======================================================================')
                # print('======================================================================')
                # print('======================================================================')

                # print('CONTENT TYPE', content_type)
                # print('RESPONSE', response)

                # Text or HTML
                if "text/" in content_type.lower():
                    # print('ITI IS TEXT')
                    res = self._mdconvert.convert_response(response)
                    # print('REEEEES')
                    # print(res)
                    self.page_title = res.title
                    # print(res.title)
                    self._set_page_content(res.text_content)
                    # print('CONTENT')
                    # print(res.text_content)
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            try:
                self.page_title = f"Error {response.status_code}"

                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                self.page_title = "Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")

    def _state(self) -> tuple[str, str]:
        header = f"Address: {self.address}\n"
        if self.page_title is not None:
            header += f"Title: {self.page_title}\n"

        current_page = self.viewport_current_page
        total_pages = len(self.viewport_pages)

        address = self.address
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == address:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                break

        header += f"Viewport position: Showing page {current_page + 1} of {total_pages}.\n"
        return (header, self.viewport)

class SearchInformationToolInput(BaseModel):
    query: str = Field(description="The web search query to perform.")
    filter_year: int | None = Field(  
        default=None,
        description="[Optional parameter]: filter the search results to a specific year (e.g., 2020).",
    )

class SearchInformationTool(BaseTool):
    name: str = "web_search"
    description: str = "Perform a web search query (think a google search) and returns the search results."
    args_schema: ArgsSchema = SearchInformationToolInput
    output_type: ClassVar[str] = "content"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def _run(self, query: str, filter_year: int | None = None) -> str:
        self.browser.visit_page(f"google: {query}", filter_year=filter_year)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content

class VisitToolInput(BaseModel):
    url: str = Field(description="The relative or absolute url of the webpage to visit.")

class VisitTool(BaseTool):
    name: str = "visit_page"
    description: str = "Visit a webpage at a given URL and return its text. Given a url to a YouTube video, this returns the transcript."
    args_schema: type[BaseModel] = VisitToolInput
    output_type: ClassVar[str] = "content"
    _browser: Any = PrivateAttr()

    def __init__(self, browser=None, **data):
        # print('initializing VisitTool')
        super().__init__(**data)
        self._browser = browser

    def _run(self, url: str) -> str:
        self._browser.visit_page(url)
        header, content = self._browser._state()
        return header.strip() + "\n=======================\n" + content

class DownloadToolInput(BaseModel):
    url: str = Field(description="The relative or absolute url of the file to be downloaded.")

class DownloadTool(BaseTool):
    name: str = "download_file"
    description: str = """
Download a file at a given URL. The file should be of this format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".png", ".docx"]
This tool then returns the path where it downloaded the file.
DO NOT use this tool for .pdf or .txt or .htm files: for these types of files use visit_page with the file url instead."""
    args_schema: type[BaseModel] = DownloadToolInput
    output_type: ClassVar[str] = "content"
    _browser: Any = PrivateAttr()

    def __init__(self, browser=None, **data):
        super().__init__(**data)
        self._browser = browser  # Maintained for compatibility with original pattern

    def _run(self, url: str) -> str:
        import requests  # Kept inside to match original implementation pattern

        # Handle arXiv PDF conversion
        if "arxiv" in url:
            url = url.replace("abs", "pdf")

        response = requests.get(url)
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type)

        # Determine file extension
        if extension and isinstance(extension, str):
            new_path = f"./downloads/file{extension}"
        else:
            new_path = "./downloads/file.object"

        # Save file
        with open(new_path, "wb") as f:
            f.write(response.content)

        # Validate allowed file types
        if "pdf" in extension or "txt" in extension or "htm" in extension:
            raise ValueError("Do not use this tool for PDF/TXT/HTML files - use visit_page instead.")

        return f"File was downloaded and saved under path {new_path}."

class ArchiveSearchToolInput(BaseModel):
    url: str = Field(description="The url you need the archive for.")
    date: str = Field(
        description="The date for the archive in 'YYYYMMDD' format (e.g., '20080627' for 27 June 2008)"
    )

class ArchiveSearchTool(BaseTool):
    name: str = "find_archived_url"
    description: str = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    args_schema: type[BaseModel] = ArchiveSearchToolInput
    output_type: ClassVar[str] = "content"
    _browser: Any = PrivateAttr()

    def __init__(self, browser=None, **data):
        # print('initializing ArchiveSearchTool')
        super().__init__(**data)
        self._browser = browser

    def _run(self, url: str, date: str) -> str:
        # Build archive API URLs
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        
        # Fetch both responses
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()

        # Check for closest match
        closest = None
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
        
        if not closest:
            raise ValueError(f"URL {url} was not archived on Wayback Machine, try a different url.")

        # Visit found archive
        target_url = closest["url"]
        self._browser.visit_page(target_url)
        header, content = self._browser._state()

        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + header.strip()
            + "\n=======================\n"
            + content
        )

class PageUpToolInput(BaseModel):
    pass  

class PageUpTool(BaseTool):
    name: str = "page_up"
    description: str = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    args_schema: type[BaseModel] = PageUpToolInput
    output_type: ClassVar[str] = "content"
    _browser: Any = PrivateAttr()

    def __init__(self, browser=None, **data):
        # print('initializing PageUpTool')
        super().__init__(**data)
        self._browser = browser

    def _run(self) -> str:
        self._browser.page_up()
        header, content = self._browser._state()
        return header.strip() + "\n=======================\n" + content

class PageDownToolInput(BaseModel):
    pass

class PageDownTool(BaseTool):
    name: str = "page_down"
    description: str = "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    args_schema: type[BaseModel] = PageDownToolInput
    output_type: ClassVar[str] = "content"
    _browser: Any = PrivateAttr()

    def __init__(self, browser=None, **data):
        # print('initializing PageDownTool')
        super().__init__(**data)
        self._browser = browser

    def _run(self) -> str:
        self._browser.page_down()
        header, content = self._browser._state()
        return header.strip() + "\n=======================\n" + content

class FinderToolInput(BaseModel):
    search_string: str = Field(
        description="The string to search for on the page. This search string supports wildcards like '*'"
    )

class FinderTool(BaseTool):
    name: str = "find_on_page_ctrl_f"
    description: str = "Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F."
    args_schema: type[BaseModel] = FinderToolInput
    output_type: ClassVar[str] = "content"
    _browser: Any = PrivateAttr()

    def __init__(self, browser=None, **data):
        # print('initializing FinderTool')
        super().__init__(**data)
        self._browser = browser

    def _run(self, search_string: str) -> str:
        find_result = self._browser.find_on_page(search_string)
        header, content = self._browser._state()

        if find_result is None:
            return (
                header.strip()
                + f"\n=======================\nThe search string '{search_string}' was not found on this page."
            )
        return header.strip() + "\n=======================\n" + content

class FindNextToolInput(BaseModel):
    pass

class FindNextTool(BaseTool):
    name: str = "find_next"
    description: str = "Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search."
    args_schema: type[BaseModel] = FindNextToolInput
    output_type: ClassVar[str] = "content"
    _browser: Any = PrivateAttr()

    def __init__(self, browser=None, **data):
        # print('initializing FindNextTool')
        super().__init__(**data)
        self._browser = browser

    def _run(self) -> str:
        find_result = self._browser.find_next()
        header, content = self._browser._state()

        if find_result is None:
            return header.strip() + "\n=======================\nThe search string was not found on this page."
        return header.strip() + "\n=======================\n" + content
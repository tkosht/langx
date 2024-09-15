from langchain.agents import load_tools
from langchain.tools.google_search.tool import GoogleSearchRun
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function


class CustomGoogleSearchAPIWrapper(GoogleSearchAPIWrapper):
    def run(self, query: str) -> str:
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return Exception("No good Google Search Result was found")
        return results


class SearchWeb(object):
    def __init__(self, n_search_results: int = 5) -> None:
        self.n_search_results = n_search_results
        self.tool: GoogleSearchRun = load_tools(["google-search"], llm=None)[0]
        self.tool.api_wrapper = CustomGoogleSearchAPIWrapper(k=n_search_results)

    def _search(self, context: SKContext) -> str:
        query = context["input"]
        # print(f"SearchWeb._search: {query=}")
        searched_results = self.tool.run(tool_input=query)

        responses = []
        for res in searched_results:
            resdic = {
                "source_url": res["link"],
                "content": res["snippet"],
            }
            responses.append(resdic)

        return str(responses)

    @sk_function(
        name="search_news",
        description="Web上のニュースや記事等を検索するときに使用します。JSON(python の dict型)を返します",
        input_description="user input or previous output",
    )
    def search_news(self, context: SKContext) -> str:
        query = context["input"]
        print(f"SearchWeb.search_news: {query=}")
        return self._search(context=context)

    @sk_function(
        name="search_weather",
        description="Web上の天気情報等を検索するときに使用します。JSON(python の dict型)を返します",
        input_description="user input or previous output",
    )
    def search_weather(self, context: SKContext) -> str:
        query = context["input"]
        print(f"SearchWeb.search_weather: {query=}")
        return self._search(context=context)

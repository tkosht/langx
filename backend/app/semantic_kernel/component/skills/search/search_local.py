from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function


class SearchLocal:
    @sk_function(
        name="search_local_files",
        description="(未実装のため使用しないこと) Webにはないローカルな情報(社内文書などのファイル群)について検索するときに使用します。JSON(python の dict型)を返します",
        input_description="user input or previous output",
    )
    def search_file(self, context: SKContext) -> str:
        query = context["input"]
        print(f"SearchLocal.search: {query=}")
        result_dict = {
            "source_file": "Not Found",
            "content": "(empty)",
        }

        return str(result_dict)

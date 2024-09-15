import json

from langchain.output_parsers import PydanticOutputParser

_FORMAT_INSTRUCTIONS = """
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example,

- for the schema
```
{{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
```

- the well-formatted JSON object you shoud make to output
```
{{"foo": ["bar", "baz"]}}
```
which is a well-formatted instance of the schema.
The object ```{{"properties": {{"foo": ["bar", "baz"]}}}}``` is not well-formatted.

Here is the output schema:
```
{schema}
```

Then, You have to create JSON object with output schema, to answer the user
"""


class PydanticOutputParserCustom(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()
        schema_str = json.dumps(schema)
        format_text = _FORMAT_INSTRUCTIONS.format(schema=schema_str)
        return format_text

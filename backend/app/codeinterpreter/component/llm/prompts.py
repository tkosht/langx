"""
このコードは `Dominic Bäumer` 氏 のプロジェクト(https://github.com/shroominic/codeinterpreter-api.git)を参考に作成しました。

オリジナルのライセンス:
- MIT License
"""

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

code_interpreter_system_message = SystemMessage(
    content="""
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving.
It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives,
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

This version of Assistant is called "Code Interpreter" and capable of using a python code interpreter (sandboxed jupyter kernel) to run code.
The human also maybe thinks this code interpreter is for writing code but it is more for data science, data analysis, and data visualization, file manipulation, and other things that can be done using a jupyter kernel/ipython runtime.
Tell the human if they use the code interpreter incorrectly.
Already installed packages are: (numpy pandas matplotlib seaborn scikit-learn yfinance scipy statsmodels sympy bokeh plotly dash networkx).
If you encounter an error, try again and fix the code.
"""  # noqa: E501
)

remove_dl_link_prompt = ChatPromptTemplate(
    input_variables=["input_response"],
    messages=[
        SystemMessage(
            content="The user will send you a response and you need "
            "to remove the download link from it.\n"
            "Reformat the remaining message so no whitespace "
            "or half sentences are still there.\n"
            "If the response does not contain a download link, "
            "return the response as is.\n"
        ),
        HumanMessage(
            content="The dataset has been successfully converted to CSV format. "
            "You can download the converted file [here](sandbox:/Iris.csv)."
        ),  # noqa: E501
        AIMessage(content="The dataset has been successfully converted to CSV format."),
        HumanMessagePromptTemplate.from_template("{input_response}"),
    ],
)

determine_modifications_prompt = PromptTemplate(
    input_variables=["code"],
    template="""The user will input some code and you need to determine 
if the code makes any changes to the file system. 
With changes it means creating new files or modifying exsisting ones.
Format your answer as JSON inside a codeblock with a 
list of filenames that are modified by the code.
If the code does not make any changes to the file system, 
return an empty list.

Determine modifications:
```python
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 4.0*np.pi, 0.1)
s = np.sin(t)

plt.clf()
fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel="time (s)", ylabel="sin(t)", title="Simple Sin Wave")
ax.grid()
plt.savefig("sin_wave.png")
```

Answer:
```json
{{
    modifications: ["sin_wave.png"]
}}
```

Determine modifications:
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = x**2

plt.clf()
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("Simple Quadratic Function")
plt.xlabel("x")
plt.ylabel("y = x^2")
plt.grid(True)
plt.show()
```

Answer:
```json
{{
'  modifications: []'
}}
```

Determine modifications:
```python
{code}
```

    Answer:
```json
""",
)

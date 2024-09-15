# flake8: noqa

tool_maker_prompt = """Please write a generic Python function to solve this type of problems using only standard python libraries. The output of the function can later be converted to the answer (option for multiple choice question). All the function should be wrapped by
```python
```"""

tool_test_prompt = """Write unit tests to verify the correctness of the function on the questions above using the following format:
```python
{parse the question into the arguments of the function}
{call the function and save the return value in a variable named `ret`}
{for multiple choice question, parse the options}
{convert the return value `ret` to the answer (if the question is a multiple choice question, convert to an option) and save it in a variable named `ans`, otherwise}
{assert ans == the provided answer (if the question is a multiple choice question, assert ans == option)}
```"""

tool_wrapper_prompt = """Success! The function is correct. We will need to summarize the function and use cases up for further use. Please extract the information from the history in the following format:

Here is a function to solve a class of problems:
```python
{the function, including necessary imports}
```

Use cases:
Question: {question (including options)}
Solution:
```python
{parse the question into the arguments of the function}
{call the function and save the return value in a variable named `ret`}
{for multiple choice question, parse the options}
{convert the return value `ret` to the answer (if the question is a multiple choice question, convert to an option) and save it in a variable named `ans`, otherwise}
```
Do this for all the questions in the verification step.
"""

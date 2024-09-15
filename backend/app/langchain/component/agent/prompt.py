# flake8: noqa
SYSTEM_MESSAGE_PREFIX = """Answer the following questions by using tools and thinking step-by-step:"""

FORMAT_INSTRUCTIONS = """
In this conversation, the way you use the tools is by specifying a json blob of the 'Action' which is defined as bellow, and also even if 'execution' in Japanese and so on, and also even if execution is on the local environment.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
And also define the successful of 'using the tools', 'execution' is the meaning no error or successful of 'Observation' which is defined as below.

# Definition of $JSON_BLOB of a 'Action'

The only values that must be in the "action" field are: {tool_names}

The $JSON_BLOB must only contain a SINGLE action, do NOT return a list of multiple actions.
Here is an example of a valid $JSON_BLOB which MUST includes '```' markers for code block in markdown format:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Note that $TOOL_NAME like "python_repl", $INPUT like "import re"
Make sure make the 1 Action very simple single and shorter, step-by-step

After you responding 'Action', you wait for the content of 'Observation' (the result of action(execution of tools))

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do by using above tools step-by-step
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final direct answer to the original input question by using the tools above
"""

SYSTEM_MESSAGE_SUFFIX = """Remember! you must always use the exact characters `Final Answer` when responding.
the phrases, Thought/Action/Observation/Final Answer, are MUST NOT be translated in Japanese.
if you found an error, must fix the error with step-by-step with 'Thought:' and 'Action:'. NEVER repeat same errors.
Let's Begin! Execute Action Step-By-Step
"""
# Note that, Especially, NEVER answer like that `I cannot directly download webpages or execute shell commands, python codes`
# you HAVE TO use the tools above to answer HUMAN directly.


# ====================================================================================================
# below is original definition
# PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
# FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
# Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
#
# The only values that should be in the "action" field are: {tool_names}
#
# The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
#
# ```
# {{{{
#   "action": $TOOL_NAME,
#   "action_input": $INPUT
# }}}}
# ```
#
# ALWAYS use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action:
# ```
# $JSON_BLOB
# ```
# Observation: the result of the action
# ... (this Thought/Action/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question"""
# SUFFIX = """Begin! Reminder to always use the exact characters `Final Answer` when responding."""

# NOTE: cf. https://github.com/hwchase17/langchain/blob/master/langchain/agents/chat/prompt.py
# Copyright (c) Harrison Chase
# Copyright (c) 2023 Takehito Oshita

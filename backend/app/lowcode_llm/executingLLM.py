from openAIWrapper import OpenAIWrapper

from app.langchain.component.agent_bot import SimpleBot


EXECUTING_LLM_PREFIX = """Executing LLM is designed to provide outstanding responses.
Executing LLM will be given a overall task as the background of the conversation between the Executing LLM and human.
When providing response, Executing LLM MUST STICTLY follow the provided standard operating procedure (SOP).
the SOP is formatted as:
'''
STEP 1: [step name][step descriptions][[[if 'condition1'][Jump to STEP]], [[if 'condition2'][Jump to STEP]], ...]
STEP 2: [step name][step descriptions][[[if 'condition1'][Jump to STEP]], [[if 'condition2'][Jump to STEP]], ...]
'''
here "[[[if 'condition1'][Jump to STEP n]], [[if 'condition2'][Jump to STEP m]], ...]" is judgmental logic. It means when you're performing this step,
and if 'condition1' is satisfied, you will perform STEP n next. If 'condition2' is satisfied, you will perform STEP m next.

Remember: 
Executing LLM is facing a real human, who does not know what SOP is. 
So, Do not show him/her the SOP steps you are following, or the process and middle results of performing the SOP. It will make him/her confused. Just response the answer.
""" # noqa

EXECUTING_LLM_SUFFIX = """
Remember: 
Executing LLM is facing a real human, who does not know what SOP is. 
So, Do not show him/her the SOP steps you are following, or the process and middle results of performing the SOP. It will make him/her confused. Just response the answer.
""" # noqa


class executingLLM:
    def __init__(self, temperature) -> None:
        self.prefix = EXECUTING_LLM_PREFIX
        self.suffix = EXECUTING_LLM_SUFFIX
        self.LLM = OpenAIWrapper(temperature)
        self.messages = [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "system", "content": self.prefix}]

        self.model_name = "gpt-3.5-turbo"
        self.temperature = temperature
        self.bot = SimpleBot()

    # def _make_query(self, messages: list):
    #     query = ""
    #     for msg in messages:
    #         query += f"{msg['role'].upper()}: {msg['content']}\n"
    #     return query

    def execute(self, current_prompt, history):
        ''' provide LLM the dialogue history and the current prompt to get response '''
        # messages = self.messages + history
        messages = list(history)
        messages.append({'role': 'user', "content": current_prompt + self.suffix})
        messages.append({'role': 'user', "content": "just now, what do I simply do?"})
        query, status = self.LLM.run(messages)
        if not status:
            return "OpenAI API error."

        print("=" * 80)
        print(f"{query=}")
        print("=" * 80)

        answer = self.bot.run(query, self.model_name, temperature_percent=self.temperature*100, max_iterations=1)
        return answer

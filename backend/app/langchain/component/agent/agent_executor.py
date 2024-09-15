import time
from typing import Any, Optional

from langchain.agents.agent import AgentExecutor
from langchain.callbacks.manager import CallbackManagerForChainRun, Callbacks
from langchain.input import get_color_mapping
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish


class CustomAgentExecutor(AgentExecutor):
    intermediate_steps: list = []
    memory: ChatMessageHistory = None

    def prep_outputs(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        return_only_outputs: bool = False,
    ) -> dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if self.memory is not None:
            intermediate_steps = outputs.pop("intermediate_steps", [])
            self.memory.save_context(inputs, outputs)
            outputs["intermediate_steps"] = intermediate_steps
            self.intermediate_steps = intermediate_steps
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping([tool.name for tool in self.tools], excluded_colors=["green"])
        self.intermediate_steps = inputs.pop("intermediate_steps", [])
        intermediate_steps: list[tuple[AgentAction, str]] = self.intermediate_steps
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(next_step_output, intermediate_steps, run_manager=run_manager)

            intermediate_steps.extend(next_step_output)
            self.check_action_repeating()

            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(tool_return, intermediate_steps)
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(self.early_stopping_method, intermediate_steps, **inputs)
        return self._return(output, intermediate_steps)

    def check_action_repeating(self, repeating_threshold: int = 3) -> None:
        if len(self.intermediate_steps) < repeating_threshold:
            return
        latest_action, _ = self.intermediate_steps[-1]
        for idx in range(2, repeating_threshold, 1):
            action, _outtext = self.intermediate_steps[-idx]
            if action.tool != latest_action.tool:
                return
            if action.tool_input != latest_action.tool_input:
                return
        raise Exception("ActionError: The Same Action Repeating")

    def run(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Run the chain as text in, text out or multiple variables, text out."""
        try:
            result = self(kwargs, callbacks=callbacks)[self.output_keys[0]]
            return result
        except Exception as e:
            print("-" * 80)
            print(f"# {e.__repr__()} at agent_executor.run()")
            if self.intermediate_steps:
                print(f"{self.intermediate_steps[-1]}")
            print("-" * 80)
            raise e


# NOTE: cf. part of https://github.com/hwchase17/langchain/blob/master/langchain/agents/agent.py AgentExecutor
# add member variable: intermediate_steps
# Copyright (c) Harrison Chase
# Copyright (c) 2023 Takehito Oshita

"""Base implementation for tools or skills."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

from langchain.agents import AgentExecutor, BaseSingleActionAgent, BaseMultiActionAgent
from langchain.agents.agent import ExceptionTool
from langchain.agents.tools import InvalidTool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import BaseTool

from spc_langchain.tools.state.tool import StateTool


class StateAgentExecutor(AgentExecutor):
    """Agent that is using tools."""

    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[BaseTool]
    """The valid tools the agent can call."""
    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    max_iterations: Optional[int] = 15
    """The maximum number of steps to take before ending the execution
    loop.    Setting to 'None' could lead to an infinite loop."""

    max_execution_time: Optional[float] = None
    """The maximum amount of wall clock time to spend in the execution
    loop.
    """
    early_stopping_method: str = "force"
    """The method to use for early stopping if the agent never
    returns `AgentFinish`. Either 'force' or 'generate'.
    `"force"` returns a string saying that it stopped because it met a
        time or iteration limit.
    `"generate"` calls the agent's LLM Chain one final time to generate
        a final answer based on the previous steps.
    """
    handle_parsing_errors: Union[
        bool, str, Callable[[OutputParserException], str]
    ] = False
    """How to handle errors raised by the agent's output parser.
    Defaults to `False`, which raises the error.
s
    If `true`, the error will be sent back to the LLM as an observation.
    If a string, the string itself will be sent to the LLM as an observation.
    If a callable function, the function will be called with the exception
     as an argument, and the result of that function will be passed to the agent
      as an observation.
    """
    state: List[Dict[str, Any]] = None
    trim_intermediate_steps: Union[
        int, Callable[[List[Tuple[AgentAction, str]]], List[Tuple[AgentAction, str]]]
    ] = -1

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, StateTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.
        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)
            if intermediate_steps:
                self.state.append(
                    {str(intermediate_steps[-1][0]): str(intermediate_steps[-1][1])}
                )
            else:
                self.state = []
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                # metadata_dict = {}
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    state=self.state,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    {
                        "requested_tool_name": agent_action.tool,
                        "available_tool_names": list(name_to_tool_map.keys()),
                    },
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

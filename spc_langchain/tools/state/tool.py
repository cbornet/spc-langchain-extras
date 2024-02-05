"""Base implementation for tools or skills."""
from __future__ import annotations

from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Type, Union

from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    CallbackManager,
    Callbacks,
)
from langchain.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
)
from langchain_core.tools import BaseTool, ToolException, SchemaAnnotationError


class StateTool(BaseTool):
    """Tool for state management. Passing intermidiate steps to tool in state variable"""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Create the definition of the new tool class."""
        super().__init_subclass__(**kwargs)

        args_schema_type = cls.__annotations__.get("args_schema", None)

        if args_schema_type is not None:
            if args_schema_type is None or args_schema_type == BaseModel:
                # Throw errors for common mis-annotations.
                # TODO: Use get_args / get_origin and fully
                # specify valid annotations.
                typehint_mandate = """
class ChildTool(BaseTool):
    ...
    args_schema: Type[BaseModel] = SchemaClass
    ..."""
                name = cls.__name__
                raise SchemaAnnotationError(
                    f"Tool definition for {name} must include valid type annotations"
                    f" for argument 'args_schema' to behave as expected.\n"
                    f"Expected annotation of 'Type[BaseModel]'"
                    f" but got '{args_schema_type}'.\n"
                    f"Expected class looks like:\n"
                    f"{typehint_mandate}"
                )

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool.
    You can provide few-shot examples as a part of the description.
    """
    args_schema: Optional[Type[BaseModel]] = None
    """Pydantic model class to validate and parse the tool's input arguments."""
    return_direct: bool = False
    """Whether to return the tool's output directly. Setting this to True means that after the tool is called, the AgentExecutor will stop looping."""
    verbose: bool = False
    """Whether to log the tool's progress."""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to be called during tool execution."""
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """Deprecated. Please use callbacks instead."""
    tags: Optional[List[str]] = None
    """Optional list of tags associated with the tool. Defaults to None
    These tags will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a tool with its use case.
    """
    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata associated with the tool. Defaults to None
    This metadata will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a tool with its use case.
    """

    handle_tool_error: Optional[
        Union[bool, str, Callable[[ToolException], str]]
    ] = False
    """Handle the content of the ToolException thrown."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        arbitrary_types_allowed = True

    def run(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        state: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool."""
        parsed_input = self._parse_input(tool_input)
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        if state:
            self.state = state
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            verbose_,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        # TODO: maybe also pass through run_manager is _run supports kwargs
        new_arg_supported = signature(self._run).parameters.get("run_manager")
        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            color=start_color,
            **kwargs,
        )
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = (
                self._run(*tool_args, run_manager=run_manager, **tool_kwargs)
                if new_arg_supported
                else self._run(*tool_args, **tool_kwargs)
            )
        except ToolException as e:
            if not self.handle_tool_error:
                run_manager.on_tool_error(e)
                raise e
            elif isinstance(self.handle_tool_error, bool):
                if e.args:
                    observation = e.args[0]
                else:
                    observation = "Tool execution error"
            elif isinstance(self.handle_tool_error, str):
                observation = self.handle_tool_error
            elif callable(self.handle_tool_error):
                observation = self.handle_tool_error(e)
            else:
                raise ValueError(
                    f"Got unexpected type of `handle_tool_error`. Expected bool, str "
                    f"or callable. Received: {self.handle_tool_error}"
                )
            run_manager.on_tool_end(
                str(observation), color="red", name=self.name, **kwargs
            )
            return observation
        except (Exception, KeyboardInterrupt) as e:
            run_manager.on_tool_error(e)
            raise e
        else:
            run_manager.on_tool_end(
                str(observation), color=color, name=self.name, **kwargs
            )
            return observation

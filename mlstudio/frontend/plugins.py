from collections.abc import Callable
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from mlstudio.backend import PipelineStep

PluginRenderer = Callable[..., PipelineStep | None]


def render_pipeline_plugins(
    feature_count: int,
    *,
    key_prefix: str,
) -> tuple[PipelineStep, ...]:
    steps: list[PipelineStep] = []
    for plugin in _discover_plugins():
        render = getattr(plugin, "render_pipeline_step", None)
        if not callable(render):
            continue
        step = render(feature_count=feature_count, key_prefix=key_prefix)
        if step is not None:
            steps.append(step)
    return tuple(steps)


def _discover_plugins() -> tuple[ModuleType, ...]:
    try:
        package = import_module("mlstudio.plugins")
    except ModuleNotFoundError:
        return ()

    plugins: list[ModuleType] = []
    for module in iter_modules(package.__path__, f"{package.__name__}."):
        try:
            plugins.append(import_module(module.name))
        except ModuleNotFoundError:
            continue
    return tuple(plugins)

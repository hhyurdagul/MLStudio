import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mlstudio.backend import PipelineStep
from mlstudio.frontend.plugins import render_pipeline_plugins


class PluginTests(unittest.TestCase):
    def test_no_installed_plugins_returns_no_pipeline_steps(self) -> None:
        with patch("mlstudio.frontend.plugins.iter_modules", return_value=[]):
            self.assertEqual(
                render_pipeline_plugins(5, key_prefix="test"),
                (),
            )

    def test_discovered_plugin_can_add_pipeline_step(self) -> None:
        step = PipelineStep("test_step", "passthrough")
        plugin = SimpleNamespace(
            render_pipeline_step=lambda **_: step,
        )
        module = SimpleNamespace(name="mlstudio.plugins.test")
        with (
            patch(
                "mlstudio.frontend.plugins.iter_modules",
                return_value=[module],
            ),
            patch(
                "mlstudio.frontend.plugins.import_module",
                side_effect=[
                    SimpleNamespace(
                        __path__=[],
                        __name__="mlstudio.plugins",
                    ),
                    plugin,
                ],
            ),
        ):
            self.assertEqual(
                render_pipeline_plugins(5, key_prefix="test"),
                (step,),
            )


if __name__ == "__main__":
    unittest.main()

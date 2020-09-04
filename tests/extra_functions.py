# From: https://stackoverflow.com/a/56048692
import pytest
from collections.abc import Mapping
from _pytest.python_api import ApproxMapping


def approx_with_mapping_support(expected, rel=None, abs=None, nan_ok=False):
    if isinstance(expected, Mapping):
        return ApproxNestedMapping(expected, rel, abs, nan_ok)
    return pytest.approx(expected, rel, abs, nan_ok)


class ApproxNestedMapping(ApproxMapping):
    def _yield_comparisons(self, actual):
        for k in self.expected.keys():
            if isinstance(actual[k], type(self.expected)):
                gen = ApproxNestedMapping(
                    self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok
                )._yield_comparisons(actual[k])
                for el in gen:
                    yield el
            else:
                yield actual[k], self.expected[k]

    def _check_type(self):
        for key, value in self.expected.items():
            if not isinstance(value, type(self.expected)):
                super()._check_type()
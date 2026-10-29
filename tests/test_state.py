import pytest

from orch.core.state import IllegalTransition, assert_transition, can_transition
from orch.domain.enums import TaskStatus as S


def test_happy_path_edges():
    for a, b in [
        (S.PENDING, S.READY),
        (S.READY, S.RUNNING),
        (S.RUNNING, S.VERIFYING),
        (S.VERIFYING, S.DONE),
        (S.VERIFYING, S.READY),
        (S.RUNNING, S.WAITING),
        (S.WAITING, S.DONE),
    ]:
        assert can_transition(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (S.PENDING, S.RUNNING),
        (S.READY, S.DONE),
        (S.WAITING, S.READY),
        (S.FAILED, S.READY),
        (S.CANCELED, S.READY),
    ],
)
def test_illegal(a, b):
    with pytest.raises(IllegalTransition):
        assert_transition(a, b)

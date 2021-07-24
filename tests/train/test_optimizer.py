import pytest
import tensorflow as tf

from tests.tests_utils import tf_equal
from transformer.train.optimizer import ScheduleLR

@pytest.fixture(scope="module")
def schedule_lr():
    return ScheduleLR(d_model=512, warmup_steps=4000, min_lr=1e-4)


# ScheduleLR
def test_init():
    schedule_lr = ScheduleLR(d_model=512, warmup_steps=100, min_lr=1e-3)
    assert schedule_lr.d_model == 512, ("schedule_lr attribute does not match"
                                        "the expected value")
    assert schedule_lr.warmup_steps == 100, ("warmup_steps attribute does not "
                                            "match the expected value")
    assert schedule_lr.min_lr == 1e-3, ("min_lr attribute does not match "
                                        "the expected value")


def test_call(schedule_lr):
    for i in range(3000, 4000):
        assert schedule_lr(i) <= schedule_lr(i+1), (
            f"schedule_lr does not increase at step {i+1}"
            )
    for i in range(4000, 5000):
        assert schedule_lr(i) >= schedule_lr(i+1), (
            f"schedule_lr does not decrease at step {i+1}"
            )
    assert schedule_lr(100000000) == 1e-4, (
        "schedule_lr limit to infinity does not match the expected value"
        )

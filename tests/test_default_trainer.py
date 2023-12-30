from blackjack.train.trainers.default_trainer import DefaultTrainer


def test_if_the_default_trainer_is_instantiated_correctly() -> None:
    error_occurred = False
    try:
        _ = DefaultTrainer(None)
    except:
        error_occurred = True

    assert not error_occurred

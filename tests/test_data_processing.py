import numpy as np

from preprocessing.data_processing import load_cleaned_data, split_dataset


def test_preprocessing_probas():
    """Asserts the probabilities all sum to 1"""
    df = load_cleaned_data()[
        ["offensive_language_proba", "hate_speech_proba", "neither_proba"]
    ]
    values = df.to_numpy()
    actual_probas = values.sum(axis=1)
    expected_probas = np.ones_like(actual_probas)
    np.testing.assert_allclose(actual_probas, expected_probas)


def test_dataset_split():
    """Asserts the split respects the proportions"""
    train, valid, test = split_dataset()
    total = len(train) + len(valid) + len(test)
    np.testing.assert_almost_equal(len(train)/total, 0.75, 3)
    np.testing.assert_almost_equal(len(valid)/total, 0.125, 3)
    np.testing.assert_almost_equal(len(test)/total, 0.125, 3)

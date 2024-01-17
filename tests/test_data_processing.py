import numpy as np

from preprocessing.data_processing import load_cleaned_data


def test_preprocessing_probas():
    """Asserts the probabilities all sum to 1"""
    df = load_cleaned_data()[
        ["offensive_language_proba", "hate_speech_proba", "neither_proba"]
    ]
    values = df.to_numpy()
    actual_probas = values.sum(axis=1)
    expected_probas = np.ones_like(actual_probas)
    np.testing.assert_allclose(actual_probas, expected_probas)

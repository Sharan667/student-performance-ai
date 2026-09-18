from src.data_preprocessing import risk_label


def test_labels():

    assert risk_label(5) == "At Risk"

    assert risk_label(12) == "Moderate"

    assert risk_label(15) == "Good"

    assert risk_label(18) == "High"

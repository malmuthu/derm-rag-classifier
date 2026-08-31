"""
Tests that the explanation never states a fact not present in the retrieved class or the classifier's own output.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "explain"))

from templates import generate_explanation

def make_fake_cases():
    return[
        {"diagnosis": "mel", "dx_type": "histo", "similarity": 0.95, "age": 55, "sex": "male", "localization": "back"},
        {"diagnosis": "mel", "dx_type": "histo", "similarity": 0.91, "age": 60, "sex": "female", "localization": "back"},
        {"diagnosis": "nv", "dx_type": "consensus", "similarity": 0.88, "age": 40, "sex": "male", "localization": "arm"},
        {"diagnosis": "mel", "dx_type": "follow_up", "similarity": 0.85, "age": 70, "sex": "female", "localization": "leg"},
        {"diagnosis": "mel", "dx_type": "histo", "similarity": 0.80, "age": 50, "sex": "male", "localization": "check"},
    ]

def test_all_mentioned_diagnoses_are_from_retrieved_cases():
    cases = make_fake_cases()
    explanation = generate_explanation("mel", 0.87, cases)

    diagnoses_in_cases = {c["diagnosis"] for c in cases}
    for cls in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        if cls in explanation:
            assert cls in diagnoses_in_cases or cls == "mel", (
                f"Explanation mentions '{cls}', which is not among the retrieved cases' diagnoses or the predicted class."
            )

def test_agreement_count_is_accurate():
    cases = make_fake_cases()
    explanation = generate_explanation("mel", 0.87, cases)

    true_agreement_count = sum(1 for c in cases if c["diagnosis"] == "mel")
    assert f"{true_agreement_count} of {len(cases)}" in explanation, (
        "Stated agreement count doesn't match the actual count of matching diagnoses among retrieved cases."
    )

def test_confidence_never_shown_as_100_percent():
    cases = make_fake_cases()
    explanation = generate_explanation("mel", 1.0, cases)

    assert "100%" not in explanation, (
        "COnfidence is displayed as 100% even with the cap in place - check logic in generate_explanation."
    )

def test_low_agreement_triggers_caution_note():
    cases = make_fake_cases()
    cases[0]["diagnosis"] = "nv"
    cases[1]["diagnosis"] = "nv"
    cases[2]["diagnosis"] = "nv"

    explanation = generate_explanation("mel", 0.6, cases)
    assert "caution" in explanation.lower(), (
        "Low agreement between prediction and retrieved cases did not trigger the caution note."
    )
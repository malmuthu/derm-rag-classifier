"""
Templated, grounded explanation generation from classifier output + retrieved similar cases.
No free-generation step.
"""

from collections import Counter

def generate_explanation(predicted_class: str, confidence: float, retrieved_cases: list[dict]) -> str:
    n = len(retrieved_cases)
    diagnoses = [c["diagnosis"] for c in retrieved_cases]
    diagnosis_counts = Counter(diagnoses)
    agreement_count = diagnosis_counts.get(predicted_class, 0)

    display_confidence = min(confidence, 0.99)

    lines = []
    lines.append(
        f"The model predicts **{predicted_class}** with an estimated confidence of "
        f"{display_confidence:.0%} (model confidence scores should be interpreted as a relative signal, not a true probability)."
    )

    lines.append(
        f"Among the {n} most visually similar cases retrieved from the case database, "
        f"{agreement_count} of {n} were also diagnosed as {predicted_class}."
    )

    if agreement_count < n / 2:
        lines.append(
            f"Note: fewer than half of the retrieved similar cases share the predicted diagnosis - this "
            f"prediction should be treated with additional caution, "
            f"and the retrieved cases below should be reviewed directly."
        )

    lines.append("\n**Most similar cases:**")
    for i, case in enumerate(retrieved_cases, 1):
        lines.append(
            f"{i}. Diagnosis: **{case['diagnosis']}** "
            f"(confirmed via {case['dx_type']}), "
            f"similarity: {case['similarity']:.2f}"
        )

    return "\n".join(lines)
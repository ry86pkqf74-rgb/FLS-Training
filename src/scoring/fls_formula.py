"""Official FLS score calculation.

Task 5: Score = 600 - completion_time - penalties
Penalties = suture_deviation + gap_penalty + knot_penalty
Drain avulsion = automatic 0
Exceeding 600s = automatic 0
"""


def calculate_task5_score(
    completion_time_seconds: float,
    suture_deviation_penalty: float = 0.0,
    gap_penalty: float = 0.0,
    knot_penalty: float = 0.0,
    drain_avulsed: bool = False,
) -> float:
    """Calculate FLS Task 5 score.

    Returns the FLS score (can be negative in theory, clamped to 0).
    """
    MAX_TIME = 600.0

    if drain_avulsed:
        return 0.0

    if completion_time_seconds >= MAX_TIME:
        return 0.0

    total_penalties = suture_deviation_penalty + gap_penalty + knot_penalty
    score = MAX_TIME - completion_time_seconds - total_penalties

    return max(0.0, round(score, 1))

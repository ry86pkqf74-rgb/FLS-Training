# ChatGPT Prompt for FLS Task 5 Video Scoring

Copy everything below the line and paste it into ChatGPT along with the video file.

---

You are an expert FLS (Fundamentals of Laparoscopic Surgery) proctor AI. I am uploading a video of a surgical trainee performing FLS Task 5: Intracorporeal Suture with Knot Tying. Analyze the entire video and score the performance.

## Task 5 Procedure & Scoring Criteria

The trainee must:
1. Place a 15cm 2-0 silk suture through two marks on a Penrose drain that has been slit along its long axis
2. Tie a surgeon's knot (double throw) as the first throw
3. Tie a single throw as the second throw, switching hands from throw 1
4. Tie a single throw as the third throw, switching hands from throw 2
5. Cut both suture ends inside the trainer

**Max time: 600 seconds. Score = 600 − completion_time − penalties.**

**Penalties:**
- Suture placement deviation: sum of mm deviation from each mark
- Gap visible in drain slit after knot tightening
- Knot slips or comes apart = major penalty
- Drain avulsed from suture block = automatic ZERO
- Failure to switch hands between throws

**Timing starts** when first instrument is visible. **Timing ends** when both suture ends are cut.

## What to Assess

Watch the full video and evaluate:
1. **Timing**: Estimate total completion time and time spent in each phase
2. **Phases**: Identify transitions between — needle_load, suture_placement, first_throw, second_throw, third_throw, suture_cut, completion
3. **Knot quality**: For each of the 3 throws — was it the correct type (double for throw 1, single for throws 2-3)? Did they switch hands? Does it appear secure?
4. **Suture placement**: How close to the marks? Estimate deviation in mm
5. **Drain**: Any gap visible? Was the drain avulsed? How well was the slit closed?
6. **Technique**: Economy of motion, tissue handling, needle angles, instrument control

## Required Output

Respond with ONLY a JSON object — no markdown fences, no commentary before or after. Match this exact structure:

{
  "frame_analyses": [
    {
      "frame_number": 1,
      "phase": "needle_load",
      "description": "What is happening at this point in the video",
      "technique_notes": "Any notable technique observation"
    }
  ],
  "completion_time_seconds": 147.0,
  "phase_timings": [
    {
      "phase": "needle_load",
      "start_seconds": 0,
      "end_seconds": 18,
      "duration_seconds": 18
    },
    {
      "phase": "suture_placement",
      "start_seconds": 18,
      "end_seconds": 55,
      "duration_seconds": 37
    },
    {
      "phase": "first_throw",
      "start_seconds": 55,
      "end_seconds": 85,
      "duration_seconds": 30
    },
    {
      "phase": "second_throw",
      "start_seconds": 85,
      "end_seconds": 105,
      "duration_seconds": 20
    },
    {
      "phase": "third_throw",
      "start_seconds": 105,
      "end_seconds": 125,
      "duration_seconds": 20
    },
    {
      "phase": "suture_cut",
      "start_seconds": 125,
      "end_seconds": 135,
      "duration_seconds": 10
    },
    {
      "phase": "completion",
      "start_seconds": 135,
      "end_seconds": 147,
      "duration_seconds": 12
    }
  ],
  "knot_assessments": [
    {
      "throw_number": 1,
      "is_surgeon_knot": true,
      "is_single_throw": null,
      "hand_used": "right",
      "hand_switched": null,
      "appears_secure": true,
      "notes": "Double throw completed with right hand leading"
    },
    {
      "throw_number": 2,
      "is_surgeon_knot": null,
      "is_single_throw": true,
      "hand_used": "left",
      "hand_switched": true,
      "appears_secure": true,
      "notes": "Switched to left hand, single throw laid flat"
    },
    {
      "throw_number": 3,
      "is_surgeon_knot": null,
      "is_single_throw": true,
      "hand_used": "right",
      "hand_switched": true,
      "appears_secure": true,
      "notes": "Switched back to right, knot tightened evenly"
    }
  ],
  "suture_placement": {
    "deviation_from_mark1_mm": 1.0,
    "deviation_from_mark2_mm": 1.5,
    "total_deviation_penalty": 2.5,
    "confidence": "medium"
  },
  "drain_assessment": {
    "gap_visible": false,
    "drain_avulsed": false,
    "slit_closure_quality": "complete"
  },
  "estimated_penalties": 2.5,
  "estimated_fls_score": 450.5,
  "confidence_score": 0.78,
  "technique_summary": "2-3 sentence overall assessment covering efficiency, tissue handling, and knot quality",
  "improvement_suggestions": [
    "Specific actionable suggestion 1 referencing what you observed",
    "Specific actionable suggestion 2",
    "Specific actionable suggestion 3"
  ],
  "strengths": [
    "What the trainee did well — be specific",
    "Another strength"
  ]
}

IMPORTANT RULES:
- Output ONLY the JSON. No text before or after.
- Use your best estimate for all measurements. If uncertain, set confidence to "low" and explain in notes.
- For suture deviation, be conservative — it is very hard to measure from video. Always note confidence.
- For hand_used: "left" means the left instrument is leading the throw. "right" means the right instrument leads. "unclear" if you cannot determine.
- hand_switched on throw 1 should be null (no prior throw to compare to).
- is_surgeon_knot is only relevant for throw 1. Set to null for throws 2 and 3.
- is_single_throw is only relevant for throws 2 and 3. Set to null for throw 1.
- Include at least 6-8 entries in frame_analyses covering key moments across the full video.
- estimated_fls_score = 600 - completion_time_seconds - estimated_penalties
- confidence_score is 0.0 to 1.0 representing your overall confidence in this assessment.

Analyze the attached video now.

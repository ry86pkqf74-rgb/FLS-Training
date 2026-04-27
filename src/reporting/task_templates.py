"""Task-specific v003 coaching templates."""
from __future__ import annotations


TASK_REPORT_TEMPLATES = {
    "task1": {
        "strength_signals": [
            "Maintains object control through mid-air transfer",
            "Uses both hands appropriately",
            "Smooth transfer rhythm",
            "Minimal collisions with pegs/board",
            "Completes outbound and return transfers",
            "Recovers minor bobbles without losing object",
        ],
        "weakness_signals": [
            "Object dropped or unrecoverable",
            "Board-assisted transfer",
            "Wrong peg/wrong side placement",
            "Excessive searching/regrasping",
            "Dominant hand over-reliance",
            "Poor handoff timing",
            "Inefficient return phase",
        ],
        "critical_errors": [
            "incomplete task",
            "lost object outside field/unrecoverable",
            "wrong transfer sequence not corrected",
        ],
        "recommended_drills": [
            "Mid-air handoff drill: transfer one object back and forth 20 times without setting it down.",
            "Peg-target accuracy drill: place each object on assigned peg with no board contact.",
            "Six-object rhythm drill: complete all six outbound transfers before return; focus on consistent cadence.",
            "Recovery-control drill: practice controlled regrasp after intentional minor bobble without object leaving field.",
        ],
        "phase_focus_rules": [
            "Emphasize object control, bimanual coordination, handoff efficiency, unrecoverable drops, and sequence completion.",
        ],
        "plain_language_summary_rules": [
            "Describe object control and transfer sequence before discussing speed.",
        ],
    },
    "task2": {
        "strength_signals": [
            "Maintains gauze tension",
            "Cuts with small controlled scissor bites",
            "Follows marked circle",
            "Uses grasper to rotate/reposition gauze safely",
            "Completes cut without detachment",
            "Avoids large line deviations",
        ],
        "weakness_signals": [
            "Deviation outside target line",
            "Large irregular cuts",
            "Loss of tension",
            "Gauze tearing",
            "Gauze detached from clamp",
            "Incomplete cut",
            "Poor scissor angle",
            "Excessive repositioning",
        ],
        "critical_errors": [
            "gauze detachment",
            "incomplete cut",
            "large off-line deviation",
            "cannot assess final cut",
        ],
        "recommended_drills": [
            "Cardinal-point accuracy drill: cut short arcs at 12, 3, 6, and 9 o'clock while measuring line deviation.",
            "Micro-bite scissor drill: use 2-3 mm scissor bites along a printed circle.",
            "Tension-control drill: maintain consistent grasper tension without tearing or tenting gauze.",
            "Half-circle segmentation drill: cut clockwise half, reset, then counterclockwise half.",
        ],
        "phase_focus_rules": [
            "Include line deviation map, worst arc, tension control, scissor bite size, and completion status when visible.",
        ],
        "plain_language_summary_rules": [
            "If deviation cannot be measured, say exact line deviation could not be determined from this camera angle.",
        ],
    },
    "task3": {
        "strength_signals": [
            "Opens loop cleanly",
            "Approaches appendage without snagging",
            "Places loop at target mark",
            "Cinches securely",
            "Cuts tail without destabilizing loop",
            "Maintains instrument control",
        ],
        "weakness_signals": [
            "Loop off mark",
            "Loose cinch",
            "Loop catches on appendage or instrument",
            "Excessive manipulation",
            "Tail cut destabilizes loop",
            "Appendage transection",
            "Incomplete loop placement",
        ],
        "critical_errors": [
            "loop not cinched",
            "loop grossly off mark",
            "appendage transection",
            "incomplete ligation",
        ],
        "recommended_drills": [
            "Loop-opening drill: open and orient the loop five times before target approach.",
            "Mark-alignment drill: place loop at the line without cinching, reset, repeat 10 times.",
            "Cinch-control drill: cinch gradually while keeping loop perpendicular to appendage.",
            "Tail-cut stability drill: cut tail while maintaining loop position and tension.",
        ],
        "phase_focus_rules": [
            "Include loop location, cinch security, appendage integrity, and tail management.",
        ],
        "plain_language_summary_rules": [
            "Prioritize final loop security and target placement over speed.",
        ],
    },
    "task4": {
        "strength_signals": [
            "Needle loaded correctly",
            "Needle passes through both marks",
            "Appropriate bite depth",
            "Smooth extracorporeal knot formation",
            "Knot pusher advances knot without loosening",
            "Slit approximates after tightening",
            "Safe tail cutting",
        ],
        "weakness_signals": [
            "Suture deviation from marks",
            "Poor needle angle",
            "Excessive drain traction",
            "Knot pusher misalignment",
            "Knot loosens during advancement",
            "Visible gap after knot",
            "Drain avulsion",
            "Knot failure",
        ],
        "critical_errors": [
            "drain avulsion",
            "knot failure",
            "visible gap after knot",
            "gross mark miss",
            "knot pusher failure preventing secure knot",
        ],
        "recommended_drills": [
            "Mark-to-mark needle driving drill: ten passes through both marks with immediate deviation measurement.",
            "Extracorporeal knot build drill: tie knot outside box, inspect structure, then deliver with pusher.",
            "Knot-pusher alignment drill: advance pusher in line with suture without levering against drain.",
            "Progressive tension drill: tighten gradually while watching slit approximation.",
        ],
        "phase_focus_rules": [
            "Include needle placement, knot pusher control, slit closure, and knot security under tension.",
        ],
        "plain_language_summary_rules": [
            "Do not let speed praise outweigh insecure knot delivery or incomplete closure.",
        ],
    },
    "task5": {
        "strength_signals": [
            "Needle orientation",
            "Accurate entry/exit through marks",
            "Proper first surgeon's knot/double throw",
            "Hand switching between throws",
            "Smooth wrapping",
            "Maintains tension without avulsion",
            "Cuts both tails appropriately",
        ],
        "weakness_signals": [
            "Mark deviation",
            "Gap visible after final knot",
            "Knot slips or comes apart",
            "First throw not double throw",
            "Missed hand switch",
            "Excessive tail length",
            "Drain trauma",
            "Cutting before knot secure",
        ],
        "critical_errors": [
            "drain avulsion",
            "knot failure",
            "visible gap after final knot",
            "failure to complete required throws",
            "failure to switch hands when required",
            "gross mark miss",
        ],
        "recommended_drills": [
            "Needle-angle drill: load needle at correct angle and drive through both marks without dragging.",
            "First-throw surgeon's-knot drill: ten double-throw first knots with deliberate tension control.",
            "Alternating-hand square-knot drill: first throw, switch hands, second throw, switch hands, third throw.",
            "Closure-before-cut drill: pause before cutting tails and verify no visible slit gap.",
            "Knot-security check drill: apply gentle opposing tension after knot completion; knot must not slip.",
        ],
        "phase_focus_rules": [
            "Prioritize closure quality, knot security, required throw sequence, hand switching, and suture placement.",
        ],
        "plain_language_summary_rules": [
            "Do not let speed praise outweigh knot failure or incomplete closure.",
        ],
    },
    "task6": {
        "strength_signals": [
            "Needle remains in view",
            "Smooth approach to ring pair",
            "Alternates central/peripheral passes as required",
            "Completes ring pairs sequentially",
            "Avoids block dislodgement",
            "Recovers needle orientation efficiently",
        ],
        "weakness_signals": [
            "Missed inner ring",
            "Missed outer ring",
            "Incorrect pass type",
            "Needle exits field of view",
            "Block dislodged",
            "Excessive reorientation",
            "Incomplete ring sequence",
        ],
        "critical_errors": [
            "needle exits field of view",
            "block dislodged",
            "failure to complete required ring sequence",
            "repeated missed inner/outer rings",
            "incorrect central/peripheral pass pattern",
        ],
        "recommended_drills": [
            "Two-ring repeat drill: repeat one ring pair until central and peripheral passes are clean.",
            "Needle-orientation reset drill: after each pass, reset needle angle before moving to next pair.",
            "Alternation callout drill: verbally call central/peripheral before each pass.",
            "No-exit field discipline drill: keep needle tip visible continuously for the entire sequence.",
            "Needle visibility discipline drill: keep the needle tip and body visible before each pass.",
        ],
        "phase_focus_rules": [
            "Include rings completed, rings missed, needle visibility, block stability, and pass-type alternation.",
        ],
        "plain_language_summary_rules": [
            "Call Task 6 a custom training task, not an official FLS manual skills task.",
        ],
    },
}

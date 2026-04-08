# LASANA:  Laparoscopic Skill Analysis and Assessment video dataset

## Background
Laparoscopic surgery refers to minimally-invasive surgical procedures that treat organs in the patient's abdomen or pelvis.
To acquire laparoscopic skills, novice surgeons practice basic tasks in a laparoscopic training box.

This dataset comprises 1270 trimmed and annotated synchronized stereo video recordings of four laparoscopic training tasks.
It is intended for the development and evaluation of methods for automatic video-based laparoscopic skill analysis.
Automatic skill analysis and assessment has the potential to support and enhance laparoscopic training, ultimately leading to better patient outcomes.

### Recorded laparoscopic training tasks
- PegTransfer: Transfer six triangular objects from the left to the right side of a pegboard, then transfer them back.
- CircleCutting: Accurately cut along a pre-marked circular path on a piece of gauze.
- BalloonResection: Carefully incise the outer balloon without puncturing the inner balloon, which is filled with water.
- SutureAndKnot: Pass a suture through a Penrose drain and close the slit with a laparoscopic knot consisting of three throws.

### Annotated video-level errors
- PegTransfer
    - object_dropped_within_fov: A triangular object is dropped within the visible field and can be retrieved.
    - object_dropped_outside_of_fov: A triangular object is dropped outside the visible field or cannot be retrieved.
- CircleCutting
    - cutting_imprecise: The cutting path deviates from the marked circle.
    - gauze_detached: The gauze becomes detached from one or more of the metal clips.
- BalloonResection
    - cutting_imprecise: The cutting path deviates from the marked line.
    - cutting_incomplete: The marked line is not fully cut.
    - balloon_opened: The inner balloon is opened, meaning that more than half of the contained water leaks out.
    - balloon_damaged: The inner balloon is damaged, but less than half of the contained water leaks out.
    - balloon_perforated: The inner balloon is perforated, resulting in only minor leakage (e.g. single drops or leakage under pressure).
- SutureAndKnot
    - needle_dropped: The suture needle is dropped.
    - suture_imprecise: The suture placement deviates from one or both of the marked dots.
    - fewer_than_three_throws: Fewer than three throws are successfully completed.
    - slit_not_closed: The knot does not close the slit in the Penrose drain properly.
    - knot_comes_apart: The knot loosens or comes apart.
    - drain_detached: The Penrose drain detaches from the Velcro strip.

### Structured skill rating
On each of four individual aspects (depth_perception, efficiency, bimanual_dexterity, tissue_handling), each recording is rated on a 5-point Likert scale.
The overall score (total global rating score, GRS) for the recording is the sum over all aspects.

The scores provided with this dataset are averaged over three independent ratings, where the scores are first Z-score normalized per rater to eliminate systematic bias.
Missing scores for the four individual aspects indicate that the recording was marked as failed by more than one rater.

Raters were instructed to mark recordings of the BalloonResection and SutureAndKnot tasks as failed if the inner balloon was opened with the first cut or, respectively, if fewer than three throws were completed. 
In this case, they did not provide scores for the four individual aspects for the recording and the normalized GRS was set to -2.5.

In total, four different raters provided the skill ratings, where two raters assessed all recordings in the dataset and two raters assessed two complementary halves of the recordings.

## Data files
Each recording is assigned a unique pseudo-English identifier (recording id).

### Video files
For each stereo video recording, the videos from the left and the right camera are stored separately, each with a spatial resolution of 960 × 540 pixels and a frame rate of 20 frames per second.
Videos are encoded using the H.264 codec and stored in the Matroska (.mkv) container format.

The video files are named after the assigned recording id and provided with the following zip-files:

- BalloonResection_left.zip: Recordings of the BalloonResection task, left camera.
- BalloonResection_right.zip: Recordings of the BalloonResection task, right camera.
- CircleCutting_left.zip: Recordings of the CircleCutting task, left camera.
- CircleCutting_right.zip: Recordings of the CircleCutting task, right camera.
- PegTransfer_left.zip: Recordings of the PegTransfer task, left camera.
- PegTransfer_right.zip: Recordings of the PegTransfer task, right camera.
- SutureAndKnot_left.zip: Recordings of the SutureAndKnot task, left camera.
- SutureAndKnot_right.zip: Recordings of the SutureAndKnot task, right camera.

### Annotation files
Video annotations are organized by task and stored in plain csv-files with semicolon delimiter, which contain one row per video recording. The first column specifies the recording id.

For each task, there is the following set of files (* stands for BalloonResection, CircleCutting, PegTransfer, or SutureAndKnot), all provided in the zip-file Annotation.zip:

- *.csv: Main annotation file with one column for each individual rating aspect (storing the aggregated normalized scores), one column for the aggregated normalized total global rating score (GRS), and one column per task-specific error (True if the error occurred in the recording). In addition, the column named `duration` stores the length of the video recording (0:mm:ss format) and the column named `frame_count` indicates the total number of frames in the video.
- *_split.csv: Default data split, where the column named `split` specifies the subset assignment (train, val, or test) of each recording.
- *_rater0.csv: Ratings assigned by rater 0 (prior to normalization), with one column for each individual rating aspect and one column for the total global rating score (GRS).
- *_rater1.csv: Ratings assigned by rater 1 (prior to normalization), with one column for each individual rating aspect and one column for the total global rating score (GRS).
- *_rater2.csv: Ratings assigned by rater 2 (prior to normalization), with one column for each individual rating aspect and one column for the total global rating score (GRS).
- *_rater3.csv: Ratings assigned by rater 3 (prior to normalization), with one column for each individual rating aspect and one column for the total global rating score (GRS).

### Miscellaneous files
- Readme.md: This Readme file.
- camera_calibration.yaml: Intrinsic parameters of the left and the right camera and their relative pose (rotation and translation).
- example_videos.zip: One example video (left camera) for each task.











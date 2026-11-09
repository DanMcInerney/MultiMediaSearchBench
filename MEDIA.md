# MultiMediaSearchBench: media sources

MMSB publishes no media. Each scenario is built from a public dataset that you download from the place its publisher hosts it,
and you are bound by the licence its publisher states. `uv run mmsb fetch --scenario all` downloads the four real scenarios'
sources from those places (README, Install) and the piles added later (below); V8's you download yourself. This page says where each source comes from, its
licence as the publisher states it, how to credit it, and what MMSB takes from it. [media/sources.jsonl](media/sources.jsonl)
holds the same facts as one record each.

| Scenario | Source | Licence | Taken |
| --- | --- | --- | --- |
| V1, `v1_security` | MEVA | CC BY 4.0 | 42 five-minute clips of three cameras and their labels (5.8 GB) |
| I3, `i3_trailcam` | Caltech Camera Traps | CDLA-Permissive 1.0 | the labels and the 4,256 photos of one location (1.9 GB) |
| A1, `a1_home_audio` | SINS | academic research only, no passing on | one day of one microphone array's minute files and the labels (4.9 GB read, 3.3 GB kept) |
| M4, `m4_samples` | FSD50K | CC BY 4.0; each clip has its own | the evaluation split's 10,231 clips and its labels (8.8 GB) |
| V7, `v7_screen` | none | | made by the code: a synthetic week of a screen, a phone, a door camera, a microphone and logs |
| V8, `v8_egolife` | EgoLife | MIT | the clips, questions and transcripts of one person's first three days (41 GB) |

## MEVA (V1)

- **From:** [mevadata.org](https://mevadata.org/). The clips are in a public Amazon S3 bucket
  (`mevadata-public-01`, release `drops-123-r13`), the labels in Kitware's
  [meva-data-repo](https://gitlab.kitware.com/meva/meva-data-repo).
- **Licence:** CC BY 4.0, as the repository's LICENSE file says.
- **Credit:** "Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research
  Projects Activity (IARPA); Corona et al., *MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection*,
  WACV 2021.
- **Taken:** the clips of cameras G336, G341 and G436 on 12 March 2018 that begin from 10:00 to before 11:15 (the 10:35 clip is
  missing for all three), and Kitware's labels of each. A camera's unbroken run of clips is joined into one MP4 by stream copy.

## Caltech Camera Traps (I3)

- **From:** [LILA BC](https://lila.science/datasets/caltech-camera-traps), in a public Google Cloud Storage bucket.
- **Licence:** Community Data License Agreement, permissive variant 1.0, as the LILA page says.
- **Credit:** Caltech Camera Traps, hosted by LILA BC; Beery, Van Horn and Perona, *Recognition in Terra Incognita*, ECCV 2018.
- **Taken:** the labels file and the 4,256 photos of location 23, with each photo's species label and capture time. The dump
  writes the capture time into each photo's EXIF, which the dataset strips.

## SINS (A1)

- **From:** the [SINS database](https://github.com/KULeuvenADVISE/SINS_database) of KU Leuven; node 7 (the living room) is
  [Zenodo record 2547315](https://zenodo.org/records/2547315).
- **Licence:** the Zenodo record says CC BY 4.0, but the `license.pdf` in the record grants a non-transferable licence "to use
  the Database for academic research purposes only", forbids transferring it to a third party without written agreement and
  commercial use without a licence, and ends after one calendar year, when copies are to be deleted. The two disagree, and MMSB
  follows the stricter. So it never copies, mirrors or hosts SINS: `mmsb fetch` downloads it from Zenodo for you, to use under
  that licence, and a dump made from it is not to be passed on.
- **Credit:** KU Leuven ADVISE and VUB ETRO as the providers; Dekkers et al., *The SINS database for detection of daily
  activities in a home environment using an acoustic sensor network*, DCASE 2017 Workshop, which the licence requires you to cite.
- **Taken:** the 1,441 one-minute files of node 7 that play on 2 February 2017, all four channels (kept as lossless FLAC), and
  the labels files `labels.csv` and `living_labels.csv`. The dump joins them into one 24-hour mono WAV.

## FSD50K (M4)

- **From:** [FSD50K](https://zenodo.org/records/4060432), through the Hugging Face mirror
  [Fhrozen/FSD50k](https://huggingface.co/datasets/Fhrozen/FSD50k), which keeps the clips as they are.
- **Licence:** CC BY 4.0 for the dataset as a whole. Each clip is its Freesound uploader's, under CC0, CC BY, CC BY-NC or CC
  Sampling+: `metadata/eval_clips_info_FSD50K.json` says which. Some are non-commercial, so use a dump of them as you would a
  non-commercial work.
- **Credit:** Fonseca, Favory, Pons, Font and Serra, *FSD50K: an open dataset of human-labeled sound events*, IEEE/ACM
  Transactions on Audio, Speech, and Language Processing, 2022.
- **Taken:** the labels, the vocabulary and the raters' judgements, and the 10,231 clips of the evaluation split; a full-size dump
  copies about 6,100 of the clips, each as the samples alone, under names that say nothing.

## EgoLife (V8)

- **From:** [EgoLife](https://egolife-ai.github.io/), on Hugging Face as
  [lmms-lab/EgoLife](https://huggingface.co/datasets/lmms-lab/EgoLife). You download it yourself (the folders `A1_JAKE`,
  `EgoLifeQA` and `EgoLifeCap`) and give its folder to `mmsb build --scenario v8_egolife --source`.
- **Licence:** MIT, as the dataset card's licence field says.
- **Credit:** the EgoLife paper, *EgoLife: Towards Egocentric Life Assistant* ([arXiv 2503.03803](https://arxiv.org/abs/2503.03803)).
- **Taken:** the clips of the first three days of `A1_JAKE`, the questions of `EgoLifeQA_A1_JAKE.json` and the transcripts of
  `EgoLifeCap`. The clips are hard-linked into the dump, not copied.

<!-- piles:start -->
## The piles of media added on 9 November 2026

`uv run mmsb fetch --scenario NAME` also fetches the sources of these scenarios (`--scenario all` is every scenario, the four
real ones first). Each source is fetched from where its publisher keeps it, at a fixed commit where it has one, into
`<sources>\<id>`; `mmsb/fetch_piles.py` is the list. A fetch that stops is continued by running it again. The maintainer decided
on 4 November 2026 that a licence is not a filter on what is used on this machine; the licence each source states is recorded
here and in [media/sources.jsonl](media/sources.jsonl) as information, and a dump made from these sources is not for publishing
without reading it. Sources marked *not scripted yet* were put on disk by the first survey and `mmsb fetch` does not make them.

| Scenario | Source | Licence | Taken |
| --- | --- | --- | --- |
| V9, `v9_lectures` | MIT OCW 14.01 (Fall 2023) | CC BY-NC-SA 4.0 | 26 lecture videos and their captions (2.9 GB) |
| A6, `a6_lecture_audio` | MIT OCW 6.0002 (Fall 2016) | CC BY-NC-SA 3.0 | 15 lectures and their captions (1.7 GB) |
| A7, `a7_podcast` | Chalk Radio (MIT OCW) | CC BY-NC-SA 4.0 | 60 episodes and their captions (3.2 GB) |
| V4, `v4_meetings` | AMI Meeting Corpus | CC BY 4.0 | 20 meetings: headset mix, corner camera where there is one, and the annotations |
| A3, `a3_meetings` | ICSI Meeting Corpus | CC BY 4.0 (its download page) | ten Bmr meetings' mixes and the annotations |
| V3, `v3_match` | SoccerTrack v2 follow-cam clips | CC BY 4.0 | three whole matches: 525 clips and their labels (8.3 GB) |
| V2, `v2_dashcam` | Australian Roads Dashcam Driving | CC BY-NC 4.0 | all 128 clips and their metadata (0.6 GB) |
| V10, `v10_home_clips` | Charades (AI2) | non-commercial; no passing on | both zips: labels and 9,848 videos at 480p (16.3 GB) |
| I7, `i7_receipts` | SROIE and CORD-v2 receipts | CC BY 4.0 (both) | the parquet files of both (0.5 + 2.3 GB) |
| I8, `i8_course_pages` | MIT OCW course PDFs | CC BY-NC-SA 4.0 | 141 PDFs of four courses (95 MB) |
| I9, `i9_nature` | iNaturalist, one observer | CC BY-NC (per photo) | 3,240 observations and their first photos (1.5 GB) |
| A8, `a8_music` | Free Music Archive (small set) | each track's artist's | the metadata zip and fma_small (8.0 GB) |
| I1, `i1_camera_roll` | ATM-Bench | CC BY-NC 4.0 | 3,759 photos, their geocoding and the questions (1.3 GB) |
| I2, `i2_screenshots` | Android in the Wild | not stated | 3,736 screenshots with OCR (1.2 GB; not scripted yet) |
| I4, `i4_site_photos` | BD3 building defects | CC BY 4.0 | 793 labelled photos (0.3 GB; not scripted yet) |
| M3, `m3_voicemail` | FTC robocalls | the README says the data itself is public domain; its `LICENSE.md` is CC BY-ND 4.0, so a dump made from it is not for passing on | 1,427 recordings with transcripts (1.15 GB) |
| A4, `a4_field` | BirdVox-full-night | CC BY 4.0 | unit 3's night of audio, the annotations of six units (0.87 GB) |

### MIT OpenCourseWare 14.01 (V9)

- **From:** [archive.org/details/mit14_01s23](https://archive.org/details/mit14_01s23), the term's recordings of the course on
  MIT OpenCourseWare (its title there: MIT 14.01 Principles of Microeconomics, Fall 2023), listed by archive.org's metadata API.
- **Licence:** CC BY-NC-SA 4.0, as the item's `licenseurl` says (`https://creativecommons.org/licenses/by-nc-sa/4.0/`).
- **Credit:** MIT OpenCourseWare, https://ocw.mit.edu/14-01F23, 14.01 Principles of Microeconomics, Fall 2023.
- **Taken:** the 26 lecture videos (360p MP4, 20.4 h) and the 26 SRT files, in one folder with the publisher's names (2.9 GB).
- **Truth:** the SRT files: OCW's captions with cue-level times, complete for speech; nothing for what is written on the board.
- **Considered:** MIT 18.642 Topics in Mathematics with Applications in Finance, Fall 2024
  ([mit18_642f24](https://archive.org/details/mit18_642f24)) was the first choice for V9 and is dropped: its 22 captions came,
  but archive.org answered HTTP 460 on every one of its 22 videos, from both of its storage nodes, from 13:38 to 14:51 EST on 9
  November 2026 (the first fetch and three retries), while other items were served. It is not fetched and not retried; the 22
  captions stay on disk in `ocw-18642`, unused.

### MIT OpenCourseWare 6.0002 (A6)

- **From:** [archive.org/details/MIT6.0002F16](https://archive.org/details/MIT6.0002F16), the term's lectures on MIT
  OpenCourseWare.
- **Licence:** CC BY-NC-SA 3.0, as the item's `licenseurl` says (`http://creativecommons.org/licenses/by-nc-sa/3.0/`).
- **Credit:** MIT OpenCourseWare, https://ocw.mit.edu, 6.0002 Introduction to Computational Thinking and Data Science, Fall
  2016.
- **Taken:** the 15 lectures (`*_300k.mp4`, 1.7 GB) and the 15 SRT files; the A6 scenario uses the sound.
- **Truth:** the SRT files, as for 14.01; the first cue of each is OCW's licence notice.

### Chalk Radio (A7)

- **From:** [archive.org/details/chalk-radio](https://archive.org/details/chalk-radio), MIT OpenCourseWare's podcast of
  interviews with its teachers, as 360p video.
- **Licence:** CC BY-NC-SA 4.0, as the item's `licenseurl` says.
- **Credit:** Chalk Radio, MIT OpenCourseWare, https://ocw.mit.edu.
- **Taken:** the 60 episodes (the trailer and seasons 1 to 7) and their 60 SRT files, named by the publisher after the episode
  and the guest.
- **Truth:** the SRT files: speech with cue times and the speakers' names at the start of turns; a podcast's opening and credits
  repeat in every episode.

### AMI Meeting Corpus (V4)

- **From:** [the Edinburgh mirror](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/) of the [AMI
  corpus](https://groups.inf.ed.ac.uk/ami/corpus/); the annotations are in its AMICorpusAnnotations folder.
- **Licence:** CC BY 4.0 for the signals, transcripts and public annotations, as LICENCE.txt in the annotation zip says.
- **Credit:** Carletta et al., *The AMI Meeting Corpus: A Pre-announcement*, MLMI 2005.
- **Taken:** the annotation files and, for the 20 meetings ES2002, ES2015, ES2016, IS1006 and TS3005 (a to d),
  `signals/<id>/audio/<id>.Mix-Headset.wav` and, where the mirror has it, `signals/<id>/video/<id>.Corner.avi`. The mirror has
  no Corner camera for IS1006 and TS3005, so those 8 meetings have audio alone; other views and documents of ES2002 that were on
  disk are kept, not fetched.
- **Truth:** human words with times for every speaker, topic segments, decisions (annotated in the ES2015, ES2016, IS1006 and
  TS3005 series, not in every meeting), dialogue acts and summaries; the role play makes the meetings' topics alike.

### ICSI Meeting Corpus (A3)

- **From:** the [Edinburgh mirror](https://groups.inf.ed.ac.uk/ami/icsi/) of the ICSI Meeting Corpus: `ICSIsignals/NXT` for the
  audio, `ICSICorpusAnnotations` for the transcripts and annotations.
- **Licence:** CC BY 4.0 for the signals, transcripts and some annotations, as its download page says.
- **Credit:** Janin et al., *The ICSI Meeting Corpus*, ICASSP 2003.
- **Taken:** both annotation zips and `signals/<id>.interaction.wav` of ten Bmr meetings: the first nine of the mirror's list in
  order (Bmr001 to Bmr010 without Bmr004) and Bmr019, with the two that were on disk (Bmr005, Bmr019).
- **Truth:** transcripts with times and the NXT layers (dialogue acts, topics, hot spots, summaries); real meetings of a speech
  research group.

### SoccerTrack v2 follow-cam (V3)

- **From:**
  [algo-dev99/soccertrack-v2-followcam-events](https://huggingface.co/datasets/algo-dev99/soccertrack-v2-followcam-events) on
  Hugging Face, read at commit `5ca71d86`; clips cut from the panoramas of [SoccerTrack
  v2](https://github.com/AtomScott/SoccerTrack-v2).
- **Licence:** CC BY 4.0, as the card says; the card asks that SoccerTrack v2 (Scott et al.) be cited.
- **Credit:** Scott et al., SoccerTrack v2; the follow-cam clips are algo-dev99's.
- **Taken:** three whole matches, the first by sorted id (117092, 117093, 118575): their 525 clips of 30 s and a label file for
  each (8.3 GB). The other seven matches and the tokens are not taken.
- **Truth:** ball events in ms per clip; derived events (`pass_received`) are noisy, a goal can be disallowed, and the camera is
  simulated, not a parent's.

### Australian Roads Dashcam Driving (V2)

- **From:**
  [qutegocentric/Australian_Roads_Dashcam_Driving](https://huggingface.co/datasets/qutegocentric/Australian_Roads_Dashcam_Driving)
  on Hugging Face, read at commit `79300a79`.
- **Licence:** CC BY-NC 4.0 (`license: cc-by-nc-4.0`), the whole of the card.
- **Credit:** QUT egocentric vision group's dataset on Hugging Face; its clips are cut from YouTube dashcam videos.
- **Taken:** all of it: 128 clips in the folders `Crash`, `Near Crash`, `Normal` and `Outliers`, and `video_metadata.csv` (0.6
  GB).
- **Truth:** a row per clip (classification, ego-centric, day or night, weather, location, hazard); nothing says when in a clip
  anything happens. Plates and faces are visible and some clips show crashes.

### Charades (V10)

- **From:** AI2's public S3 bucket: `https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/` ([project
  page](https://allenai.org/plato/charades/)).
- **Licence:** a non-commercial licence (`Charades/license.txt` in the annotation zip): academic, non-profit and government
  research, and evaluation; **it does not grant the right to distribute the data to a third party or to modify and publicly
  release it**. `mmsb fetch` downloads it from AI2 for you; never copy it where others can fetch it from.
- **Credit:** Sigurdsson et al., *Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding*, ECCV 2016.
- **Taken:** `Charades.zip` (the labels) and `Charades_v1_480.zip` (9,848 videos at 480p), as zips; a scenario reads the videos
  out of the zip.
- **Truth:** timed action spans of 157 classes (a video's `actions` field), the script, scene and objects; crowd workers acted a
  script in their own homes, so the clips are staged and the labels are not exhaustive.

### SROIE receipts (I7)

- **From:** [jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) on Hugging Face, read at commit
  `bffe40c2`: the ICDAR 2019 receipt competition's data.
- **Licence:** CC BY 4.0, as the card and its NOTICE say.
- **Credit:** Huang et al., *ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction*.
- **Taken:** the repository as it is: two parquet files (987 receipts, 0.5 GB), kept as fetched, and its README, LICENSE and
  NOTICE.
- **Truth:** company, date, address and total of each receipt and its words with boxes; a cashier's first name or a member
  number can be on a receipt.

### CORD v2 receipts (I7)

- **From:** [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) on Hugging Face, read at commit
  `7f0115a4`.
- **Licence:** CC BY 4.0, as the card says.
- **Credit:** Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*, NeurIPS 2019 Document Intelligence
  Workshop.
- **Taken:** all six parquet files (1,000 receipts, 2.3 GB), kept as fetched.
- **Truth:** menu lines (name, count, price), subtotal and total of each receipt; Indonesian item names, some truncated; most
  receipts have no date.

### OCW course pages (I8)

- **From:** [ocw.mit.edu](https://ocw.mit.edu/): the PDFs linked from each course's `download` page, for 6.006 (Spring 2020),
  9.00SC (Fall 2011), 7.016 (Fall 2018) and 5.111SC (Fall 2014).
- **Licence:** CC BY-NC-SA 4.0, as each course page declares.
- **Credit:** MIT OpenCourseWare, https://ocw.mit.edu, with each course's title and term.
- **Taken:** all of the PDFs those pages link (141 files, 95 MB), one folder per course (`<course>/<name>.pdf`).
- **Truth:** the text of the PDFs themselves (read by the scenario); nothing else labels them.

### iNaturalist observer (I9)

- **From:** the [iNaturalist API](https://api.inaturalist.org/v1/docs/) for one observer's observations, and iNaturalist's
  open-data bucket for the photos (`inaturalist-open-data.s3.amazonaws.com`).
- **Licence:** each photo's own: the bucket holds only CC0, CC BY and CC BY-NC photos, and the 3,240 kept are all CC BY-NC (the
  `photo_license` field). Use the pile as a non-commercial work.
- **Credit:** the observer's photos and observations, from iNaturalist: credit the observer as the photo's licence asks.
- **Taken:** research-grade observations of user 5669063 from 2022-01-01 to 2025-12-31 that have photos of open licence:
  `observations.jsonl` (no location, no user name) and the first photo of each (`photos/<photo id>.<ext>`).
- **Truth:** the community identification (taxon name and rank), the observation date and time; identifications can be coarse or
  wrong, and hands or faces can be in a photo.

### Free Music Archive (A8)

- **From:** the FMA's [store](https://os.unil.cloud.switch.ch/fma/) at the Swiss research network (SWITCH), described at
  [github.com/mdeff/fma](https://github.com/mdeff/fma).
- **Licence:** each track is licensed by its artist (the `license` column of `tracks.csv` says how: many are CC BY-NC-SA or CC
  BY-NC); use the pile as a non-commercial work.
- **Credit:** Defferrard, Benzi, Vandergheynst and Bresson, *FMA: A Dataset for Music Analysis*, ISMIR 2017; and each artist.
- **Taken:** `fma_metadata.zip` and `fma_small.zip` as zips (8.0 GB); a scenario reads the clips out of the zip.
- **Truth:** per track: genres (uploader-assigned), duration, the date it was added and recorded, artist, title, language,
  licence; no instruments and no times inside a clip.

### ATM-Bench (I1)

- **From:** [Jingbiao/ATM-Bench](https://huggingface.co/datasets/Jingbiao/ATM-Bench) on Hugging Face, read at commit `78e826dc`.
- **Licence:** CC BY-NC 4.0, as the card says.
- **Credit:** the ATM-Bench paper (arXiv 2603.01990).
- **Taken:** the photos, their geocoding, the question files and the processed-memory files (7,528 files, 1.31 GB): the folder
  the first survey made, which `mmsb fetch` reproduces and finds complete.
- **Truth:** 1,013 questions with evidence ids (a third cite emails or videos that are not taken); the file names carry each
  photo's capture second.

### Android in the Wild (I2)

- **From:** [Android in the Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild), in
  Google's public bucket.
- **Licence:** not stated on the README page that was read.
- **Credit:** Rawles et al., *Android in the Wild: A Large-Scale Dataset for Android Device Control*.
- **Taken:** **not scripted yet**: 3,736 screenshots of shards 1 to 12 as PNG with their OCR text and boxes (1.19 GB), made by
  the first survey; `mmsb fetch` says so and does not make them.
- **Truth:** OCR text and element boxes per screenshot, the goal and device per episode.

### BD3 building defects (I4)

- **From:** [chandrabhuma/building_defect_vqa](https://huggingface.co/datasets/chandrabhuma/building_defect_vqa) on Hugging
  Face.
- **Licence:** CC BY 4.0, as the card says.
- **Credit:** the BD3 dataset, chandrabhuma/building_defect_vqa.
- **Taken:** **not scripted yet**: the test split's 793 photos and their labels (0.32 GB), written out of the parquet by the
  first survey.
- **Truth:** one defect class per photo.

### Robocall audio (M3)

- **From:** [wspr-ncsu/robocall-audio-dataset](https://github.com/wspr-ncsu/robocall-audio-dataset), as the zip of its main
  branch.
- **Licence:** the README says the data itself is in the public domain and describes CC BY-ND.
- **Credit:** the robocall-audio-dataset of the wspr-ncsu group.
- **Taken:** the recordings (1,427), `metadata.csv`, the case PDFs, README and LICENSE: 1,461 files (1.15 GB); the zip is not
  kept.
- **Truth:** a Whisper transcript per call (model output, not checked) and the FTC case.

### BirdVox-full-night (A4)

- **From:** [Zenodo record 1205569](https://zenodo.org/records/1205569).
- **Licence:** CC BY 4.0, as the record says.
- **Credit:** Lostanlen, Salamon, Farnsworth, Kelling and Bello, *BirdVox-full-night: a dataset for avian flight call detection
  in continuous recordings*.
- **Taken:** the FLAC audio of unit 3, the call annotations of six units, GPS, UTC start times and README (10 files, 0.87 GB),
  as the first survey made the folder.
- **Truth:** timed flight calls of passing migrants (time and frequency, no species).

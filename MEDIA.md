# MultiMediaSearchBench: media sources

MMSB publishes no media. Each scenario is built from a public dataset that you download from the place its publisher hosts it,
and you are bound by the licence its publisher states. `uv run mmsb fetch --scenario all` downloads the four real scenarios'
sources from those places (README, Install); V8's you download yourself. This page says where each source comes from, its
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

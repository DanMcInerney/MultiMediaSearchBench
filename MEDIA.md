# MultiMediaSearchBench: media sources

Status: found and partly downloaded on 4 November 2026. Every scenario in [DESIGN.md](DESIGN.md) section 5 now has
real media to start from, except the private ones (X1, X2). 79.4 GB of starter media is on disk in
`D:\bench-data\sources\`. Several scenarios fall short of the dump DESIGN asks for, and those gaps need Dan: see
"Decisions for Dan". How the search was done, and the checks on its result, follow the decisions.

**In use (6 November 2026).** Four sources now make the built scenarios (DESIGN.md section 5.7), each read where it is and
copied into the dump, never written to:
- `meva` (V1): the three visible cameras of 12 March 2018, 10:00 to 11:15, with Kitware's two official label sets; the infrared
  cameras and the night clip are not used, as they have no official labels.
- `caltech-camera-traps` (I3): location 23, all 4,256 photos, with the species labels. The labels' times drop AM and PM, so the
  hour after midnight reads as noon; the photos print the right one, and I3 reads it.
- `sins` (A1): node 7's whole day of 2 February 2017, one channel, with the living room's and the home's activity labels; the
  FSD50K needles are not mixed in yet.
- `fsd50k` (M4): the evaluation split with its human-verified labels; a dump leaves out the sounds easily taken for the ones it
  asks about.

## Summary

| ID | Scenario | Use | On disk now | Truth it carries | Biggest gap |
| --- | --- | --- | --- | --- | --- |
| V1 | Doorbell or security export | `meva` | 3 overlapping cameras × 75 min, the 3 thermal cameras for the same window, the one night slot; 8.0 GB | activity spans to the frame (707 in the starter), camera overlap table | staged: a mock town with hired actors; at most ~13 h per camera, so no 48-hour export; one 5-minute night slot; no doorbell camera; package events nearly absent |
| V2 | Dashcam road trip | `ia-dashcam-bristol`, `comma2k19` | 62.6 min of real 3-minute chapters; 58 min of comma2k19; a real SD-card sample; 8.0 GB | per-second GPS and UTC track inside each Bristol file; GPS, pose and CAN for comma2k19 | no timed event labels (cut-ins, exits, signs) |
| V3 | Kid's recital or match | `yt-recital-beehive-2026`, `yt-match-nvysl-u12-dragons`, `yt-concert-deltavista-band-2026` | a 1 h 56 min recital, an 80 min U12 match, a 96 min 4K band concert; 10.1 GB | 5 goal times from the uploader; nothing else | no 3-hour 4K phone file; YouTube re-encodes, not originals; dances and shirt numbers need a person, or SoccerTrack v2 |
| V4 | Lecture, meeting or conference | `ami`, `yt-conf-icml2024-mechinterp` | AMI ES2002a-d (2.4 h, 6 camera views, slides, whiteboard); an 8 h 20 min workshop stream; 5.5 GB | AMI: slide times, whiteboard strokes, `stand_whiteboard` spans, human transcripts with forced-alignment word times; workshop: programme in wall-clock time | the workshop programme must be aligned to the stream; AMI's server is slow (0.06-0.5 MB/s) |
| V5 | Lab or robot camera | `droid`, `vifailback` | 148 DROID runs × 3 cameras, one scene (60 succeed); ViFailback labels for 5,202 runs and 33 videos; 2.0 GB | success flag per run; ViFailback: failure type and frame | no hours-long lab camera; DROID says that a run failed, not when |
| V6 | Bird-feeder or wildlife camera | `feedercam-birdsuntold` | one 11.6 h dawn-to-dusk feeder day; 180 day logs; 5.5 GB | per species: visits, first and last seen (minute), from the project's detector and BirdNET | labels are model-made; an on-screen overlay shows visit counts and must be cropped |
| V7 | Developer screen | the copied synthetic week | not part of this search | by construction | AgentNet (DESIGN 11.3) was not checked |
| V8 | Egocentric days | EgoLife | already on disk before this search: `D:\bench-data\egolife` (41 GB) | EgoLifeQA | none new |
| (new) | Kitchen | `epic-kitchens` | participant P14, 8.7 min; 2.0 GB | 98 timed sound events, 87 timed actions | far too little for a dump; the server is slow and drops connections |
| I1 | Phone camera roll | `atm-bench`, then `disbench`, `camroll-yfcc20` | one person's 3,759 phone photos over 4 years (resized to about 1000×750, EXIF time and GPS kept), with 1,013 questions in that person's voice; two DISBench users; 2.6 GB | ATM-Bench: questions with evidence photo ids, written and checked by trained annotators; DISBench: 122 find-all queries | no open roll of 10-30k with HEIC, bursts and WhatsApp copies; no owner-written questions; YFCC copies have no EXIF |
| I2 | Screenshots folder | `android-in-the-wild`, `rico` | 3,736 real phone screenshots (as PNGs written from the dataset's pixel arrays); 3,000 app screens; 2.5 GB | OCR text and element boxes per screenshot; view hierarchies | not one person's folder; "wifi password" questions need planted screens or Dan's own |
| I3 | Trail camera SD card | `caltech-camera-traps` | camera location 23, 6 months, 4,256 images (80.5% empty); 2.0 GB | species and capture time to the second for every image | single frames, not bursts; the labels' times drop AM and PM (the hour after midnight reads as noon; the photos print the right one) |
| I4 | Home inspection or insurance | `home-inspection-mobilehome`, `bd3-building-defects`, `houses-dataset` | 124 phone photos of one damaged home; 793 labelled defect close-ups; 2,140 room photos; 1.3 GB | a damage list by room (names no photos); one class per defect photo | no open 2k inspection folder; a person must tag which photos show each damage item |
| I5 | Family archive scans | `ia-family-photos-0487`, `fgnet` | 1,376 scans of one family's prints; 1,002 scans of 82 people at ages 0-69; 0.9 GB | FG-NET: identity and age per scan | no open 8k family archive with people labelled |
| I6 | Construction time-lapse | `anu-physics-webcam` | 1,918 hourly 4K frames of a building site, 2018-2021 (9 zero-byte, as in the source); 1.5 GB | time per frame; crane up 23-24 Aug 2019 and down on 20 Feb 2021, read by eye | hourly, not every 5 minutes; crane times need a person to confirm |
| A1 | Home while away | `sins` Node 7, with `fsd50k` needles | one whole real day of a living room, 23.98 h, no file missing; 3.3 GB | activity spans to the ms (away, visit, TV, vacuum); sound needles to be mixed in | SINS has no alarm, doorbell or dog; no second home recording found |
| A2 | Overnight recording | `sins` Node 10, `apsaa`, `donateacry` | one real bedroom night, 8.75 h; one hospital sleep-room night, 7.9 h; 457 cleaned and 1,128 raw infant-cry uploads from a phone app; 1.5 GB | sleeping span; snore and apnea events (APSAA); cry reason per clip | no open sleep-talk or baby-monitor night |
| A3 | Meeting or interview | `ami`, `icsi` | AMI ES2002a-d and ES2016a-d headset mixes (4.5 h); 2 ICSI meetings (2.2 h); counted under V4 and 0.3 GB | human transcripts with forced-alignment word times; decision, topic, dialogue-act and agreement layers | no long interview-tape collection with timed transcripts verified |
| A4 | Field recorder | `birdvox-full-night`, `birdvox-full-season`, soundscape sets, `anuraset` | three consecutive real nights of one recorder, 32.8 h; 2 labelled soundscape hours; all soundscape labels; 2.9 GB | 9,113 timed flight calls on the middle night; species boxes in the soundscape sets | no multi-night recording with owl or frog labels; owls and frogs must be mixed in |
| A5 | Car or machine | `mimii` | slider id_06: 89 abnormal and 111 normal 10-s clips; 0.5 GB | normal or abnormal per clip | no long real drive or factory recording; joined clips leave seams |
| M1 | Voice memos | `ia-voice-memos`, `voicenotes-rosehill`, `mtg-qbh` | one uploader's 121 phone-recorder files (10 h, not yet listened to); 108 app voice notes and 19 scripted notes (each also denoised); 118 hummed melodies; 1.3 GB | app transcripts (model) for the notes; song per hum | no real memos with human transcripts |
| M2 | Messaging voice notes | `whatsapp-ptbr-bernardoai`, `voicenotes-rosehill` | 198 WhatsApp clips (0.5 h); 0.1 GB | original text and a model transcript | no open set near 1,000; the real fit, EMOVOME, is gated |
| M3 | Voicemails | `ia-industry-voicemail`, `robocall-ftc`, `ia-markpines-answering`, `minds14` | 256 real business voicemails (5.7 h); 1,427 robocalls (10.1 h); 3 home answering-machine tapes; 563 phone-quality clips; 1.7 GB | model transcripts (robocalls, tapes); none for the business voicemails | no human transcripts |
| M4 | Sample library | `fsd50k`, `esc50` | FSD50K's whole eval split (10,231 clips) and 3,572 dev clips from 30 household classes; ESC-50; 12.2 GB | human-verified labels, 200 classes; mirror checked file by file against Zenodo | no effect labels ("with reverb") |
| M5 | Recorder on a schedule | `anuraset` | 183 real one-minute files from a 1-in-15-minutes schedule; all labels; 1.0 GB | per-file species activity; timed calls | frogs, not owls; 183 files is under the dump's 3,000; the best fit (Dja AudioMoth) needs a person in a browser |
| M6 | Motion clips with sound | `xd-violence`, `mavd-traffic` | 221 videos with their own sound (7.5 h); 7 street-camera recordings; 3.0 GB | frame spans of explosions, crashes, fights; timed traffic sounds | XD-Violence is partly film and news; no open doorbell clips with sound |
| X1 | Scripted real day | Dan's own recording | nothing to find | the script | private |
| X2 | Whole phone dump | Dan's phone | nothing to find | owner-written | private; `atm-bench`'s 533 videos (not downloaded) would make a public stand-in |
| L | Large-file track | re-muxed from the scenarios | real big single files are now on disk: a 6.1 GB 4K concert, a 5.5 GB 11.6-hour feeder day, a 3.0 GB 8-hour stream, 11-hour FLAC nights, a 24-hour room day as 1,441 one-minute files | as in the source scenarios | none new |

## Decisions for Dan

1. **Gated sources.** Each needs an account, an access request or a signed agreement, which no agent made:
   - SoccerTrack v2 (V3): jersey number per player per frame, and ball actions. Gated on Hugging Face, but its
     [project page](https://atomscott.github.io/SoccerTrack-v2/) also links a public Google Drive folder that lists
     without sign-in. Whether its files download was not tested.
   - EMOVOME (M2): 999 real WhatsApp voice messages with human transcripts; a signed agreement.
   - VIRAT 2.0 (V1): a Kitware usage agreement.
   - RIUADA (I4): a Zenodo access request.
   - DreamCatcher (sleep talk), ICSD and CryCeleb2023 (A2): Hugging Face gates.
   - ivrit-ai crowd WhatsApp (M2): a Hugging Face gate. Common Voice Spontaneous Speech (M2): a Mozilla Data
     Collective account.
   - PhotoBench (I1): images only on request.
   - Roboflow and Kaggle building-damage sets (I4), and Flickr Commons through the Flickr API (I5): accounts.
   - LDC Voicemail (M3), NVIDIA CASCADE (V2), Freesound (M4): a licence or an account.
2. **Sources a person should try in a browser.** Each refused a script or showed a human check, which no agent got
   around:
   - Dja AudioMoth (M5): a real recorder schedule with experts' species per file, 176 GB. The API needs a token, and
     the web download shows an "are you human" check. Probably no account is needed in a browser.
   - DEW (I5): 1 million dated Flickr photos, 1930-1999, behind a bot check.
   - PSG-Audio (A2): a JavaScript page with no file URL a script could find.
   - The WSU drill run-to-failure set (A5): its CDN refused scripts.
3. **Scenario shapes the open media cannot give.** Each needs a design change or Dan's own recordings:
   - V1: MEVA has at most ~13 h per camera. Build the 1 h and 4 h rungs, or a multi-day export with the gaps a real
     recorder export has (camera set 3-336 in full is ~36 camera-hours, ~82 GB).
   - V3: no 3-hour 4K phone file. Join two shows or matches, or use the large-file rung.
   - I1: no open single-person roll of 10-30k. Build it from ATM-Bench and DISBench users, with HEIC, bursts and
     WhatsApp copies made by the builder, or use Dan's phone.
   - I6: hourly frames, not every 5 minutes.
   - A2: no open sleep talk. Dan's own nights, or a scripted recording.
   - A5: no long real base. Dan records a long drive or a running appliance with a fault at a known time, or MIMII
     clips are joined with seams.
   - M1-M3: no real memos, notes or voicemails with human transcripts. Either a person checks model transcripts on the
     lines that questions rest on, or the scripted sets in DESIGN 7.1 are recorded.
4. **Jobs for a person** (each short): mark the dances and soloists in the recital (V3); annotate timed driving events
   such as cut-ins, exits and signs, or keep V2's questions to what GPS and speed can check; align the workshop's slots
   to its stream (V4); tag which mobile-home photos show each damage item (I4); say who is who in the family scans
   (I5); confirm the crane hours (I6); listen to `ia-voice-memos` to learn what is in it (M1). Also decide whether
   model-made labels count as truth: Birds Untold's detector and BirdNET logs (V6), and Whisper or app transcripts
   (M1-M3).
5. **How real is real enough:**
   - MEVA (V1) is real security cameras in a mock town (the Muscatatuck Urban Training Center), with more than 100
     hired actors in scripted and unscripted scenarios.
   - ATM-Bench (I1) photos are resized to about 1000×750 with camera make and model removed. Its questions were
     written by trained annotators in the owner's voice, not by the owner (DESIGN 7.1, way 3).
   - YouTube re-encodes for V3, V4 and V6. Originals exist on the Internet Archive at 0.7 MB/s.
   - XD-Violence for M6 is partly film and news.
   - The trail camera writes midnight as 12:xx (I3): keep it or fix it.
   - SINS has a DC offset of half of full scale (A1, A2): keep it or remove it. It must be removed before mixing
     needles either way.
6. **Scope:** whether to add a Kitchen scenario from EPIC-KITCHENS (CC BY-NC; 8.7 minutes on disk so far), and whether
   to fetch ATM-Bench's 533 videos as a public stand-in for X2.
7. **Two items left out on purpose (V1).** On the Internet Archive, Ring doorbell clips from the Karen Read homicide
   case, at a named home, and "Found CCTV DVR Footage", recovered recordings that may come from a place that serves
   children. They were not downloaded. Recommended: keep them out.
8. **Flickr originals (I1).** Full-size YFCC photos with real EXIF are open, but Flickr answered HTTP 429 after 128
   files and the search stopped. A slow fetch that honours the rate limit is possible if wanted.
9. **Privacy.** Much of this media is about identifiable private people:
   - children: the recital, the U12 match (its description names players), the school concert, the DonateACry infants;
   - named third parties: callers on the Mark Pines answering-machine tapes;
   - people in litigation: the business voicemails come from discovery in opioid-sales cases;
   - an unknown uploader's voice memos (`ia-voice-memos`);
   - an identifiable family (`ia-family-photos-0487`);
   - homes and places: the mobile-home photos and ATM-Bench's photos keep GPS in their EXIF;
   - health: APSAA is hospital patients' sleep recordings.

   Recommended: none of these goes into a published dump or question set; questions are never built around a child's
   or a third party's name; and GPS is stripped from anything that leaves this machine.
10. **Publishing and licences.** SINS's record says CC BY 4.0, but its licence PDF says academic use only, with no
    transfer to third parties; APSAA forbids redistribution; several YouTube and Internet Archive items state no
    licence. Under Dan's decision they are used on this machine. If MMSB publishes dumps, these must stay out of the
    published ones.

## How this was done

- **Searched:** the open web, for each scenario in DESIGN section 5. There were five searches, one per media shape:
  dataset video, web video, images, long audio and short audio. Each scenario section below was written by the search
  that covered it, so "I" there is that search. Every source was opened by it, unless the row says "read" or
  "unverified".
- **Licences:** on 4 November 2026 Dan decided that licences are not a filter: media on the open internet may be used.
  The licence each source states is recorded as information only (`licence_stated` in
  [media/sources.jsonl](media/sources.jsonl)). This covers use on this machine. Whether dumps built from these sources
  may be published is still open (DESIGN section 13, question 8).
- **Not done:** no accounts were made, no sign-ins, no agreements or access requests, no torrents, and no bot checks
  or rate limits were worked around. Sources behind any of these are listed as `gated` or `blocked` and were not
  downloaded.
- **Downloaded:** a starter set for most scenarios, 79.4 GB in all, in `D:\bench-data\sources\<id>\`, one folder per
  source. A starter is enough media for a scenario's smallest rung (about an hour, or a few hundred to a few thousand
  files), plus all of the source's labels and metadata. Each folder has a `MANIFEST.jsonl` with each file's URL, bytes,
  SHA-256 and a probe (duration, size, codec), made by opening and decoding the file.
- **Full size later:** each source's `fetch_full` in [media/sources.jsonl](media/sources.jsonl) fetches the rest. Some
  of those commands use small range-request scripts, kept in `D:\bench-data\tools\fetch\<search>\` (`zipget.py`,
  `sins_fetch.py`, `rzget.py`, `rsql2.py`, `iadl.py`, `s3list.py` and others). The searches' own notes are in
  `D:\bench-data\sources\_notes\`.
- **Reviewed:** an independent reviewer checked this file, the source list and the downloads against the sources on 4
  November 2026, and the corrections it found are applied here.

## Checks on the result

Run on 4 November 2026, after the searches finished:

- **Downloads against manifests:** 57 source folders, 79.4 GB, 63,630 files. Every file on disk is listed in its
  folder's manifest, and every listed file exists. Its size matches, and its SHA-256 was recomputed and matches.
- **Media decode:** each search probed its own files. A second check decoded 444 media files picked at random (up to
  12 per folder), reading a frame or a second near the start and near the end. All decoded except 5 of the 7 files
  of the `ia-70mai-card` sample, a recovered SD card whose time-lapse, parking and fragment files are corrupt on the
  card itself. That is the point of that sample (V2).
- **Links:** this file holds 127 links. The 119 that are not YouTube were fetched after the last edit, and three failed:
  - AMOS (I6) does not resolve;
  - the WSU drill set (A5) refuses scripts;
  - MEVA's GitLab page refuses scripts, although its annotation archive downloads (HTTP 206).

  The first two are listed as blocked below. The reviewer found the 8 YouTube links public (through YouTube's oEmbed).
- **Gating:** the reviewer found that all 15 downloaded Hugging Face repos report `gated: false` in the Hugging Face
  API.
- **Not checked by a person:** the truth claims that come from reading media (the I6 crane hours, the V6 overlay,
  what is said in the memos and voicemails).

## 5.1 Big video

### V1 Doorbell or security export

**Use:** `meva`. MEVA is footage from fixed security cameras at the Muscatatuck Urban Training Center, a mock town
(the file names say school, hospital, admin and bus station), recorded in 2018 with more than 100 hired actors in
scripted and unscripted scenarios ([mevadata.org](https://mevadata.org/)). The cameras, codecs and clip cuts are real;
the people are partly acting. It has activity spans
to the frame and a table of which cameras share a view. So "when", "count", "absent" and "which camera saw it first"
questions can be checked by rule. To build the dump, join each camera's consecutive 5-minute clips into 1-hour MP4
exports. The recorder made the cuts and consecutive clips abut, so the join restores the original stream; I tested a
stream-copy join with ffmpeg. MEVA cannot give 3 cameras × 48 h: each camera has only 11-13 h in the whole release.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [MEVA](https://mevadata.org/) ([bucket](https://mevadata-public-01.s3.amazonaws.com/?list-type=2&prefix=drops-123-r13/), [annotations](https://gitlab.kitware.com/meva/meva-data-repo)) | 28 fixed cameras (24 visible-light, 1920×1080 H.264 in the clips I probed; 4 thermal IR, 352×240) in 5-min AVI clips named by local start and end time; 7 days, March 2018, mostly 09:00-18:00 | direct (S3 lists without signing; GitLab archive) | 3,956 clips, 475.9 GB (about 330 camera-hours at 5 minutes a clip); whole bucket 630.8 GB | activity spans per clip (37 types, start and end frame at 30 fps, actor boxes): 769 clips evaluation-level, 1,443 training-level; camera sets and frame offsets between cameras | 3 overlapping cameras (G336, G341, G436) for 2018-03-12 10:00-11:15, every clip annotated, plus 3 IR cameras for the same window, plus the one night clip of all six, plus G336's two latest May clips (21:05-21:15): 75 min per camera, 3.75 h visible-light and 3.75 h IR, plus 10 min from May; 7.96 GB with all annotation files (repo tar.gz) |
| [VIRAT 2.0](https://data.kitware.com/#collection/56f56db28d777f753209ba9f) | the earlier surveillance set from the same programme | gated: a usage agreement on the collection | 111.5 GB | activity annotations (a snapshot sits openly in the MEVA bucket) | none |
| [Ring doorbell clips, Karen Read case](https://archive.org/details/1MeadowsRingfootage) | a real Ring export: 283 motion clips of 60-67 s, MP4 and WMV (some blurred), named by epoch milliseconds | direct | 5.64 GB | none (time from the name) | none: evidence from a homicide case at a named home, with an abusive item description; needs a decision |
| [Found CCTV DVR Footage](https://archive.org/details/FoundCCTVDVRFootage2015) | recovered deleted DVR recordings, MPEG-2 VOB in 1 GB pieces | direct | 319 GB | none | none: privacy (the uploader says it may come from a premises serving children) |

What I checked in MEVA:
- **Clips.** 5-minute AVI, H.264, no audio, named like `2018-03-12.10-15-02.10-20-02.school.G336.r13.avi` in folders
  `<date>/<hour>/`. A camera's consecutive clips abut: each ends at the next one's start, and probes give 300.0 s. There
  are holes: all cameras lack 10:35-10:40 on 2018-03-12.
- **Length per camera.** By clip count, G336 has about 12.9 h, and G341 and G436 about 11.6 h each, spread over 7 days.
  The longest unbroken run of any
  camera is 2.8 h (2018-03-07 09:11-12:00). A real 1-hour export can be rebuilt; a 24-hour one cannot.
- **Overlap.** `metadata/meva-clip-camera-and-time-table.txt` groups cameras into sets that "share sufficient
  field-of-view that some degree of frame-level synchronization is possible", with frame offsets. Set 3-336 is G336
  (school) with G341 and G436 (hospital). The other sets are pairs: G299+G330, G326+G329, G340+G424, G419+G420,
  G505+G506, G508+G509, G300+G639.
- **Night and IR.** In the March release the only night footage is the 23:55-00:00 clip of 2018-03-11, one clip per
  camera, unannotated. I looked at G436's first frame: the camera is in monochrome night mode. The four thermal cameras
  are G474, G475, G476 and G479; 9 IR clips are annotated. The May drop (`drop-5-mevid`) has isolated clips up to 21:15;
  I
  downloaded G336's 21:05 and 21:10 clips and they are still daylight, so May adds no night.
- **Annotations.** KPF YAML per clip (`*.activities.yml`, `*.geom.yml`, `*.types.yml`). Times are frame numbers within
  the
  clip at 30 fps; the clip name gives local wall time. The starter window has 707 activity spans of 24 types on the 3
  cameras, mostly vehicles starting, stopping and turning, doors and trunks, people talking and using phones. Across the
  whole release, `person_puts_down_object` and `person_picks_up_object` each have about 5,000 spans,
  `person_steals_object` 5 and `person_abandons_package` 1 (the two `status.csv` files). Cross-camera person identities
  are in `mevid-annotations/` (MEVID; I took only its 12 MB annotation zip).

**Gaps:**
- 48 h per camera does not exist in MEVA. Dan must choose: build the 1 h and 4 h rungs only, or make a
  multi-day export of all of a camera set's footage, with the gaps a real NVR export has. Set 3-336 in full is about
  36 camera-hours over 7 days, 82 GB.
- Only one 5-minute slot of night in the March release. MEVA has no doorbell-style camera: its cameras are high, wide
  views of roads and car parks.
- I found no open, long doorbell export that is free of the problems above. On the Internet Archive I searched titles
  only, so this is a gap, not proof that none exists.
- A lead I did not check: the "Daily Signals" monthly bundles by the Bristol dashcam uploader (below) list
  "fixed-location
  camera footage (Ring)" among their contents, but they are 70-223 GB tar.gz files, so the contents could not be listed.

### V2 Dashcam road trip

**Use:** `ia-dashcam-bristol` for the footage and `comma2k19` for truth checked against GPS and CAN data. The Bristol
archive is a real phone dashcam's unedited output: 1080p chapters of about 3 minutes for nearly every day of 2026. Date,
time, speed and GPS are burned into each frame, and the file carries a per-second GPS and UTC-time track. A dump is a
run of real chapters; several hours a day are available. comma2k19 has one-minute segments with per-frame GPS time,
pose, CAN speed and steering, so "where" and "how fast" can be graded. But each comma2k19 drive keeps only the highway
part: the median is 12 minutes and the longest 31.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [Dashcam Archive, Bristol VA](https://archive.org/details/1775401073) ("A Year of Driving", William Carty) | continuous daily dashcam from the "iOS DashCam Recorder" app: HEVC 1920×1080 30 fps MOV chapters of about 180.5 s, named by Unix time; no audio; CC BY 4.0 | direct | ~2.5-4 TB est. (176 daily items, 10-30 GB each by the items I listed; the 2026-09-09 item is 1.37 GB, so days vary) | per-second GPS track in the MOV (UTC date and time, lat, lon, altitude, speed, heading, acceleration); the same burned into the picture; no event labels | 2026-04-05 08:14-09:17: the day's first unbroken run, 21 chapters, 62.6 min, 5.28 GB (MD5s match IA's list) |
| [comma2k19](https://huggingface.co/datasets/commaai/comma2k19) (Hugging Face mirror) | 2,035 one-minute segments of a highway commute in 2 cars; raw HEVC 1164×874 at 20 fps, no container | direct (zip members fetched by HTTP range) | 94.6 GB | per-frame GPS time and pose, CAN speed and steering, IMU; no events | 2 drives, 58 one-minute segments (58 min, unbroken within each drive), with GPS, pose and CAN logs; 2.67 GB |
| [70mai card dump](https://archive.org/details/no-20251130-200134-000385-f) | a real 70mai SD card from 2 days, described as "Corrupt Recovered": 3-min "NO" chapters (front 4K HEVC with sound, rear 1080p HEVC), plus parking, event and time-lapse files and 563 recovered fragments | direct | 181 GB; only the 191 NO files are playable by IA's list (7.7 h front plus rear, 75.6 GB) | none | 7 sample files, 60 MB: the NO pair opens; the parking, time-lapse and fragment samples do not |
| [NVIDIA CASCADE](https://huggingface.co/datasets/nvidia/cascade) | driving clips with labels for the ego car's and other agents' actions | gated: Hugging Face licence agreement | unverified | lane changes and other actions (from a search summary; unverified) | none |

What I checked in the Bristol archive: the first chapter of 2026-04-05 is `DR260405/1775391244.MOV`. Its name is
2026-04-05 12:14:04 UTC, the burned-in clock reads 08:14:05 local, and the first GPS record reads 12:14:05 UTC.
That day has unbroken runs of 63, 39, 48, 30 and 124 minutes (from the item file list). Chapter names step by
181-182 s, but chapters last 180.5 s (the first 181.5 s), so consecutive chapters leave gaps of about 0.5-1.5 s, as a
real recorder's do. The run's last chapter is 147 s. Each MOV carries three timed-metadata tracks; the first holds
one record per second: `A,050426,121405,36.627220,-82.175256,...` (UTC date and time, latitude, longitude, then
acceleration, speed, altitude and heading, as far as I can read the fields). The first frame shows a wet two-lane road
under cloud.

**Gaps:**
- No open driving set with timed cut-in, exit or sign labels was found. Event questions need a person to annotate them,
  or must be questions that check against GPS and speed ("where did we stop", "how fast were we going at 10:42").
- comma2k19 has no whole trips. "8 h in 4 GB chapters" must come from the Bristol archive (or the 70mai card).
- The wrong camera clock must be made by the builder. In the Bristol files the burned-in clock and the GPS track are
  right. The builder can shift file names and container times; the burned-in time then shows the truth, which is how a
  real wrong clock looks only if the overlay is wrong too.

### V3 Kid's recital or match

**Use:** `yt-recital-beehive-2026` for the recital and `yt-match-nvysl-u12-dragons` for the match;
`yt-concert-deltavista-band-2026` when the rung needs one big 4K file. The recital is a parent's handheld recording of a
whole show from the balcony (zoom changes, a railing in view, blackouts between dances). The match is a parent's fixed
wide-angle sideline recording whose uploader listed the goals with times and shirt numbers. The concert is a 96-minute
school band concert at 2160p from the seats, 6.1 GB in one file. None is 3 hours. A 3-hour dump needs two shows of one
studio, or several matches of one team (this parent posted about ten more), split or joined as a phone would; the
large-file rung can re-mux them into one file.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [yt-recital-beehive-2026](https://www.youtube.com/watch?v=4w1ev4cyrkI) | Beehive dance studio 2026 recital, whole show, from the seats ("Not my music") | direct (yt-dlp) | 1.84 GB at 1080p60 DASH | none: no chapters, no description, no programme found | 1 file, 1080p60 H.264, 6968 s (see manifest) |
| [yt-match-nvysl-u12-dragons](https://www.youtube.com/watch?v=BML_2W2j50Y) | NVYSL U12 Dragons vs Alum Rock FC, full match, sideline camera | direct (yt-dlp) | 2.15 GB at 1080p60 | 5 goal events with mm:ss and shirt numbers (2 marked unconfirmed by the uploader) | 1 file, 2.15 GB, 1080p60, 4806 s |
| [yt-concert-deltavista-band-2026](https://www.youtube.com/watch?v=smvqfYYY8GM) | Delta Vista Middle School band spring concert 2026, 4K from the seats, heads of the audience in the frame | direct (yt-dlp) | 6.12 GB at 2160p30 VP9 | none: no chapters, no description; ensemble changes and applause mark the pieces | 1 file, 3840x2160 VP9, 5782 s |
| [soccertrack-v2](https://atomscott.github.io/SoccerTrack-v2/) | 10 full amateur university matches, 4K panoramic rig | gated on Hugging Face; the project page also links a public Google Drive folder (lists without sign-in; file downloads untested) | ~900 min (size not stated) | jersey number per player per frame; 12 ball-action classes with ms times | not downloaded |
| [ia-recital-vaibhaba-2024](https://archive.org/details/bedford-Vaibhaba_School_Dance_Recital_2024) | Vaibhaba School Dance Recital 2024, community TV, one original 1080p .mov | direct (0.7 MB/s today) | 6.07 GB, 2 h 42 min | none | item metadata only; the .mov was started, too slow, stopped and deleted |
| [ia-youthsoccer-easton-2022](https://archive.org/details/ecatma-Easton_Youth_Soccer_Under_the_lights_10_29_22) | 4 h 13 min of youth games in one file, community TV | direct | 3.98 GB, 720p | none | not downloaded |

Rejected: recital videos sold by studios (paid and edited); "full concert" uploads by artists (not personal media);
Little League World Series games (broadcast, not from the stands).

**Gaps:** No full-length 4K phone recording of 3 hours was found; the only 4K find is the 96-minute band concert. No
recital with a findable printed programme was found, so "find my daughter's solo" needs a person to mark dances and
soloists (the blackouts make boundaries easy). In the match, players are a few pixels tall: shirt numbers cannot be
read,
so "every time number 7 touched the ball" has no truth here and probably no answer from the pixels. SoccerTrack v2 has
exactly that truth. It is gated on Hugging Face, but its project page also links a public Google Drive folder, which
lists without sign-in; whether its files download was not tested.

### V4 Lecture, meeting or conference recording

**Use:** `ami` for the meeting dump and `yt-conf-icml2024-mechinterp` for the livestream. AMI gives room video from six
cameras, slides named by the seconds they were on screen, whiteboard strokes with timestamps, and per-person movement
spans including `stand_whiteboard`, so "who drew on the whiteboard" and "when was this slide shown" are known from the
data. The ICML workshop stream is one 8 h 20 min file of talks with slides on screen and a published programme.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [ami](https://groups.inf.ed.ac.uk/ami/corpus/) | AMI Meeting Corpus, 100 h of design meetings; 6 cameras, headset, lapel and array microphones, slides, whiteboard | direct (Apache listing at [/AMICorpusMirror/amicorpus/](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/)) | ~1 TB with full-res video (estimate) | slide JPEGs named `<start>__<end>.jpg` (s); whiteboard strokes XML; movement spans incl. `stand_whiteboard`; human transcripts with forced-alignment word times (10 ms); topics; dialogue acts | ES2002a-d (2.39 h): low-res Corner and Overhead for all four; all six views and full-res Corner for ES2002a; every slide, whiteboard and shared-doc file |
| [yt-conf-icml2024-mechinterp](https://www.youtube.com/watch?v=-gTkBM42uOY) | ICML 2024 Mechanistic Interpretability Workshop, full-day stream, Vienna | direct (yt-dlp) | 2.49 GB video 720p25 + 0.49 GB audio | programme with wall-clock slots and paper order ([schedule](https://icml2024mi.pages.dev/), [orals and posters](https://icml2024mi.pages.dev/posters)); YouTube ASR captions (model output) | 720p file (29996 s), captions VTT, both schedule pages |
| [m3av](https://huggingface.co/datasets/CHHHH/M3AV_v1.0) | M3AV, ~367 h of academic lectures; transcripts with word timestamps and slide OCR | direct (HF, not gated) | 2.02 GB annotations + audio parts; video only as YouTube links | transcripts with word times; slide OCR (show times unverified) | not downloaded |

AMI's six views per meeting: Closeup1-4 (one face each), Corner (the whiteboard and projector end), Overhead (the whole
table). Low-res AVIs are 352x288 MPEG-4 at 25 fps; `_orig.avi` files are 720x576 MPEG-4 at 25 fps; audio is 16 kHz
mono PCM WAV (all probed). ES2002 movement annotations hold 8 `stand_whiteboard` spans (ES2002a: all four people,
ES2002c: three, ES2002d: one).

**Gaps:** The workshop programme is in wall-clock time and the stream starts before the 09:00 opening: the captions put
"please welcome David" at 0:24:27 and the 12:00 panel's start near 3:16:13, so slots must be aligned by a person or from
the captions. Nobody has labelled what each slide shows. A 20-40 h meeting dump needs 35-70 AMI meetings (they average
about 35 minutes); the Edinburgh mirror gave 0.06-0.5 MB/s per connection today, so plan hours for a large pull.

### V5 Lab or robot camera

**Use:** `droid` for near-identical runs, and `vifailback` for runs where the moment of failure is labelled. DROID's raw
release has per-episode MP4s from 3 cameras, in separate `success/` and `failure/` folders. In one kitchen scene, one
operator recorded 148 runs in 77 minutes, 146 of them the same lid task; 60 succeeded. The dump is these 148 episode
folders, or
their MP4s renamed as a lab would keep them. "Find a run where the lid task failed" is checked against the success flag.
Telling one failure from another needs a person, or labels like ViFailback's.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [DROID raw 1.0.1](https://droid-dataset.github.io/) ([GCS listing](https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=robotics/droid_raw/1.0.1/)) | Franka teleoperation episodes from 13 labs, 3 ZED cameras (wrist and 2 external) | direct | 8.7 TB (5.6 TB without stereo and SVO, per the DROID site) | per episode: success flag, task category, scene, lab, operator, time; up to 3 instructions (47 of the 60 successes in the starter; no failures) | lab REAL, scene 3827924651, 2023-06-23: 148 episodes (60 success, 88 failure) × 3 mono MP4, plus `trajectory.h5` and metadata; 1.78 GB |
| [ViFailback](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset) | 5,202 runs of an ALOHA two-arm robot, mostly failures; the card says 100 tasks, the repo has 109 task folders | direct | 492.2 GB of HDF5 (5,202 files; repo 512.9 GB, per the Hugging Face API) | per run: failed or not, failed subtask, failure type, frame of the failure and of the correction | annotations for all 5,202 runs (109 task files); videos of one task (cut_sausage: 33 runs, 31 failed, 640×480 25 fps, 7.1 min); 0.18 GB |

What I checked in DROID:
- Each episode folder has `metadata_<lab>+<user>+<time>.json`, `trajectory.h5` and `recordings/MP4/<serial>.mp4`.
  The metadata gives lab, user, building, scene_id, success, task category, trajectory length and camera serials. The
  folder also holds `-stereo.mp4` files and `recordings/SVO/`, which I did not take.
- The MP4s are 1280×720 H.264 with one frame per 15 Hz control step, but tagged 60 fps, so they play 4× fast. Real time
  is 4× the container duration: 32 minutes per camera in the starter.
- Instructions are in `aggregated-annotations-030724.json` (50,092 episodes). In the starter, 47 of the 60 successes
  have
  them and no failure does.
- On disk, the colons in the bucket's folder names are replaced by "-".

**Gaps:**
- I found no long continuous lab or robot camera recording (hours). The starter's DROID episodes last 3.5-29 s, and its
  148 runs span 77 minutes with pauses between them.
- DROID's failure flag says that a run failed, not how or when.
- ViFailback's light videos show one camera. Its HDF5 files hold all cameras but are about 150 MB per run.

### V6 Bird-feeder or wildlife camera

**Use:** `feedercam-birdsuntold`. One 11.6 h dawn-to-dusk broadcast (2026-10-22) from a fixed feeder camera, with the
project's own per-day log by species. On that day the log lists 11 species at the feeder, including Downy Woodpecker
(15 visits, first 13:28) and Red-bellied Woodpecker (5 visits, 12:29-15:21), so "when did the woodpecker visit?" has a
logged answer. Visits fill about two of the 11.6 hours. The log is coarse (first and last seen to the minute, presence
per hour) and is made by the project's YOLO detector and BirdNET, not by people. The project archives one broadcast a
day:
165 distinct stream URLs in 180 day files, so the dump can grow to days or weeks of the same camera.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [feedercam-birdsuntold](https://birdsuntold.pages.dev/) | Lake Ridge VA feeder cam, [2026-10-22 broadcast](https://www.youtube.com/watch?v=J0vNtRGrBpc), one per day | direct | 5.53 GB at 720p with audio, per day | per day and species: visits, first and last seen (minute), hourly presence; BirdNET the same by sound; on-screen overlay with a running visit count | 720p file of the 11.6 h day (41932 s); 180 day logs and 164 audio-day logs |
| [feedercam-normhughes-ok](https://www.youtube.com/watch?v=juKuApEg3u4) | Oklahoma backyard feeders, 11.9 h archive with night visitors | direct | 7.71 GB at 720p with audio | sparse chat sightings with wall-clock times | not downloaded |
| [feedercam-northtexas](https://www.youtube.com/watch?v=iWhk3NvkwfQ) | North Texas feeder, 4K, 11.9 h, night IR, burned-in clock | direct | 6.04 GB at 720p, 47.6 GB at 4K, with audio | none useful | not downloaded |
| [nestcam-dorset-hobby](https://www.youtube.com/watch?v=1n3Xc-zKOXY) | hobby (falcon) nest cam, 10.7 h, daily archives | direct | 7.35 GB at 720p with audio | 3 uploader chapters (not inspected) | not downloaded |

Rejected: "Cat TV" 8-12 h feeder videos (edited compilations, often with music); Six Star Adventures' 6 h "feeder cam"
(Ring motion clips joined, by its own description); Cornell Lab and Explore.org cams (live only; no long archives
found);
streams with background music (Hamilton Pecking Order).

**Gaps:** Birds Untold's picture carries an overlay (top right: clock, "Scanning / Waiting for visitors", visit count,
species seen today) that gives answers away; it must be cropped or blurred before use. Reading the overlay over time is
also the cheapest way to get visit-level times. The project's own files disagree: the "today" file fetched for
2026-10-22 said 216 visits and 13 species, the day file says 98 visits and 11 species, so treat the counts as soft and
spot-check. No fixed-camera wildlife video dataset with per-visit, human-made species labels was found that downloads
without an account (SSW60's feeder clips: unverified, and short clips, not continuous).

### V7 Developer screen

**Use:** the copied synthetic week (DESIGN section 10), with its real footage. Nothing new was searched for V7. The two
real hours of a developer's screen that calibrated the generator are already on disk
(`D:\bench-data\screen-calibration`, `screen-calib-hourA`, `screen-calib-hourB`).

**Gaps:** AgentNet's screen recordings (DESIGN 11.3) were not checked in this search.

### V8 Egocentric days

**Use:** EgoLife, already downloaded for the copied EgoLifeQA run: `D:\bench-data\egolife` (41 GB). Nothing new was
searched for V8.

### Kitchen (a possible new scenario)

**Use:** `epic-kitchens`. This is real head-mounted kitchen video with sound. EPIC-SOUNDS gives timed sound events (44
classes, start and stop to the millisecond), and EPIC-100 gives timed actions. So "when did the tap run" or "how many
times was the fridge opened" can be checked by rule.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [EPIC-KITCHENS-100](https://epic-kitchens.github.io/) with [EPIC-SOUNDS](https://github.com/epic-kitchens/epic-sounds-annotations) | about 100 h of head-mounted kitchen video (DESIGN 11.3), 1080p at 59.94 or 50 fps with audio (`EPIC_100_video_info.csv`); CC BY-NC 4.0 | direct (data.bris.ac.uk over HTTPS) | ~1.3 TB est. (P01_01 alone is 6.2 GB for 27.5 min) | 78.4k timed sound events; narrated action segments with start and stop | participant P14, videos P14_01-P14_07: 8.7 min, 1.98 GB, 98 sound events and 87 action segments; all EPIC-SOUNDS and EPIC-100 annotation CSVs |

**Gaps:** data.bris.ac.uk was slow and dropped connections, so I took only P14's seven short videos. P14 is the
participant with the least labelled footage (9 videos, 19.6 min, 197 sound events); P14_08 and P14_09 (10.9 min,
2.46 GB) are left. Eight minutes is far below a dump: a kitchen scenario would need several participants. I did not
check the extension videos (2020).

## 5.2 Many images

### I1 Phone camera roll

**Use:** `atm-bench` first, then `disbench`. ATM-Bench is one real person's phone camera roll: 3,759 photos (and 533
videos) over about 4 years (2021-2025), named `YYYYMMDD_HHMMSS.jpg`, with EXIF capture time and GPS kept (device make
and
model stripped), and 1,013 questions in the owner's voice, written by five trained annotators and checked by a second
one
([paper](https://arxiv.org/abs/2603.01990), appendix B.2); 632 of them are answered by photos or videos named in
`evidence_ids`
("Which hotel did I stay in on the first day of my trip to Kent?"). It is the closest open match to the scenario, at the
2k rung, but its photos are resized (about 1000×750, camera
make and model removed) and its questions are not the owner's own (DESIGN 7.1, way 3).
DISBench adds 57 Flickr users' streams of 1,285-1,995 photos each, with capture time, GPS, address and camera per photo
and 122 verified find-all queries; pooled with other users they reach 10-30k as background. The YFCC copies have no
EXIF, so a builder writes it back from the metadata.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [atm-bench](https://huggingface.co/datasets/Jingbiao/ATM-Bench) | one person's photos, videos and email summaries over ~4 years (UK and China); photos ~375 KB JPEG, about 1000×750, EXIF time and GPS kept, identifying EXIF stripped, sensitive regions blacked out by the authors | direct (HF, not gated) | 3.32 GB repo (photos, videos, emails) | `atm-bench.json`: 1,013 questions by trained annotators (514 open, 360 number, 139 list recall), each with `evidence_ids` (photo/video names or email ids); 632 point at media only; also NIAH variants | 3,759 images, 7,528 files, 1.31 GB; all 3,759 photos (none of the videos), questions, geocoding cache |
| [disbench](https://huggingface.co/datasets/RUC-NLPIR/DISBench) | 57 YFCC100M users, 109,467 photos. Checked: users have 1,285-1,995 photos over up to 20 years; cameras are mostly DSLRs and compacts. Phone-like users: `70554294@N00` (Sony Ericsson K790a phone, 1,986), `47554402@N00` (iPhone 4 and Samsung S860, 1,586), `35032604@N00` (iPad), `47642109@N04` (Galaxy Note). Dates include junk (1886, 1970, 2027) | direct (HF, not gated) | 14.47 GB | `queries.jsonl`: 122 queries, target photo ids (avg 3.84); per photo time to the second, GPS, address, camera | 3,572 images, 3,689 files, 0.52 GB; metadata, ids and queries for all 57 users; all images of 70554294@N00 and 47554402@N00 |
| [yfcc100m](https://registry.opendata.aws/multimedia-commons/) (Multimedia Commons S3) | the YFCC100M images and a 65.6 GB SQLite of all metadata in a public bucket that lists without signing; no Yahoo agreement was asked for | direct; Flickr originals: blocked (HTTP 429 after 128 files) | 99.2M images; metadata 65.6 GB | none; metadata: user, date taken, camera, GPS, tags, licence | 128 images, 129 files, 75 MB; 128 Flickr originals with EXIF; full YFCC rows for 3,572 photos of the two DISBench users |
| [camroll-yfcc20](https://huggingface.co/datasets/thaoshibe/camroll-yfcc20) | 20 YFCC100M users, 15,607 photos (505-979 each), a profile photo of each owner | direct (HF, not gated) | 2.29 GB | 40 episodic multiple-choice questions per user with `image_ref` (the photos holding the answer), 10 persona questions; option a is correct | 959 images, 1,001 files, 0.71 GB; questions, metadata and profile photo of all 20 users; all images of 12992900_N00 |
| [dalle-mini/YFCC100M_OpenAI_subset](https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset) | metadata (and images) of the 14.8M-photo CLIP subset, with `uid`, `datetaken`, `capturedevice` and `secretoriginal` | direct (HF, not gated) | 1.02 GB metadata; 1.9 TB images | none | none (metadata split into many small files; not fetched in the time) |
| [PhotoBench](https://huggingface.co/datasets/SorrowTea/PhotoBench) | 3 real personal albums (~1,070, 1,470 and 1,050 photos) with 1,187 queries | gated: raw images "available upon request"; the card's Drive link returned 404 | unknown | queries with answers for the 300 validation queries | none (repo listed: queries only, 240 KB) |

**Settled today:**
- **Streams can be rebuilt per photo, not per user.** The S3 SQLite (`tools/etc/yfcc100m_dataset.sql`) has `photoid` as
  its
  primary key and no index on user. Reading it by range requests fetched all 3,572 rows of the two phone users in about
  5
  minutes. Finding every photo of a user would mean scanning 65.6 GB once. DISBench's users are capped at 2,000 by whole
  photosets (its paper, appendix C.4), so their full streams are longer than what DISBench ships.
- **EXIF does not survive in the open copies.** The S3 images and DISBench's `images.zip` are byte-identical (404 of 404
  compared), at most 500 px, with no EXIF. camroll's users `12992900_N00` (702 MB) and `12754005_N00` (561 MB) are
  larger; `12992900_N00` is up to 2048x1536 and keeps camera EXIF and DateTimeOriginal (in 146 of 200 probed files incl.
  the 20 small profile photos), the other 18 users are 500 px without EXIF. ATM-Bench keeps capture time and GPS.
- **Flickr originals keep everything** (full resolution, camera, both DateTimeOriginal and an edit time, real crops),
  and
  `https://live.staticflickr.com/<server>/<id>_<secretoriginal>_o.jpg` serves them without an account. Flickr answered
  HTTP
  429 after 128 files (4 connections, 0.2 s apart), so I stopped. The 128 (user `47554402@N00`: iPhone, iPhone 3G,
  Samsung
  S860) are kept as reference originals.
- **One heavy user's stream:** not settled. DISBench's users each had at least 2,000 photos; the true maximum needs the
  full metadata scanned (the OpenAI-subset metadata gives a lower bound per user if counted).

**Gaps:** no open set is one person's phone roll at 10-30k with HEIC, bursts and WhatsApp copies; those must be made by
the builder (convert some JPEGs to HEIC, strip dates from "WhatsApp" copies, duplicate bursts) or come from Dan's phone.
Original-resolution photos with EXIF are only reachable through Flickr, which rate-limits. Other open personal photo
collections published whole (one person's years of phone photos) were not found; camroll and DISBench are the closest.

### I2 Screenshots folder

**Use:** `android-in-the-wild` and `rico`. AitW screenshots are whole phone screens (status bar,
notification icons, home screen, Chrome, Gmail, settings), with OCR text and element boxes per screenshot, so "find the
screenshot that shows X" can be checked by rule against the OCR. The dataset stores them as pixel arrays in
TFRecords, so the PNGs on disk were written from those arrays and are not files as a phone saves them. Rico adds 66,261
screens from 9.7K apps with their full
view hierarchies (all on-screen text). A dump is 5k of these renamed `Screenshot_20250612-181502.png` style. Neither is
one person's folder: no chats, wifi passwords or photos of the web, so the "wifi password" kind of question needs
needles (screens with planted text) or Dan's own screenshots.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [rico](https://www.interactionmining.org/archive/rico) | 66,261 unique screens from 9.7K Android apps, captured in 2017 by crowd workers; per-screen files at `.../rico_dataset_v0.1/combined/<n>.jpg/.json` | direct (GCS) | 6.47 GB (`unique_uis.tar.gz`) | view hierarchy text, classes and bounds per screen (text on 172 of 200 checked); semantic annotations (25 component types); app and category per screen | 3,000 images, 6,003 files, 1.26 GB; 3,000 random screens with hierarchies, semantic_annotations.zip, ui and app CSVs |
| [android-in-the-wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild) | 715K episodes, 5.69M phone screenshots (Pixel 3-6 and others) from raters doing tasks; status bar, notifications, home screens, Chrome, settings, test accounts; stored at reduced resolution (270-720 px wide in 200 probed) | direct (GCS) | `general/` 13.6 GB; all ~several hundred GB (estimate) | OCR text, element types and boxes per screenshot; goal and action | 3,736 images, 3,749 files, 1.19 GB; general/ shards 1-12 (of 321) and every screenshot in them as PNG, with OCR text, element boxes, goal and device in screenshots_labels.jsonl |
| [wave-ui](https://huggingface.co/datasets/agentsea/wave-ui) | 80K element rows over web, mobile and desktop screenshots | direct (HF, not gated) | 3.40 GB | OCR, name, type and purpose per element | none |
| [ShowUI-desktop](https://huggingface.co/datasets/showlab/ShowUI-desktop) | desktop screenshots from OmniAct with grounding rows | direct (HF, not gated) | 0.52 GB | element boxes and instructions | none |
| MIRIX ScreenshotVQA ([paper](https://arxiv.org/abs/2507.07957)) | one person's 5-18K screenshots with owner-written questions | blocked: not released | | | rejected: not available |

**Gaps:** no open set found that is one person's real screenshots folder. Desktop screenshots in bulk (wave-ui,
ShowUI-desktop, `biglab/webui-all` at 550 GB) were checked as not gated but not downloaded.

### I3 Trail camera SD card

**Use:** `caltech-camera-traps`, location 23. One camera's images in time order are an SD card. Location 23 has 12,375
images over 13 months, 83% empty, with deer on 63 days, coyote, rabbit and bobcat, and night frames in grayscale
infrared. Every image's species label and capture time are in the metadata, so "which nights did deer come?" and "was
there ever a bear?" (no bear class exists) have rule-checked answers.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [caltech-camera-traps](https://lila.science/datasets/caltech-camera-traps) | 243,100 images, 140 cameras, SW United States, 2010-2015; 2048x1494 JPEG, UUID names, no EXIF | direct (GCS; also Azure and AWS) | ~105 GB | checked: `date_captured` to the second for all 243,100 images, `location` for all, species per image (21 classes + empty), ~66K boxes. Empty by label: 51.7% overall (LILA's page says ~70%), 97.6% at location 26, 83% at 23 | 4,256 images, 4,260 files, 1.95 GB; location 23, 2013-09-01 to 2014-02-28, plus all label files |
| [lila-snapshot-serengeti](https://lila.science/datasets/snapshot-serengeti) | ~7.1M images, ~76% empty (LILA page), seasons 1-11 | direct | ~6 TB | checked for season 1 (411,414 images): species per image from sequence labels, datetime to the second, 168 locations, frame number in sequence; 82.4% empty by label | 0 images, 1 file, 19 MB; season 1 metadata only |
| [lila-wellington](https://lila.science/datasets/wellingtoncameratraps) | 270,450 images, 187 locations in New Zealand, ~17% empty (LILA page) | direct | 176 GB | checked: species per image; datetime to the minute for all 270,450; 215 location ids (the page says 187 locations); 16.9% empty by label | 0 images, 1 file, 9 MB; metadata only |
| CCT20 ([same page](https://lila.science/datasets/caltech-camera-traps)) | the 20-location benchmark subset, images at max 1024 px | direct | 6 GB | same labels | annotations only (`eccv_18_annotations.tar.gz`); rejected as the base: downsized, and drops the empty frames the scenario needs |

**Starter:** location 23, 1 September 2013 to 28 February 2014: 4,256 images, 80.5% empty, deer on 35 days, coyote 292,
deer 240, rabbit 162 images. Night frames by the clock (19:00-06:00) were all grayscale in a 107-image sample.

**Clock mess (checked):** no image at location 23 or 26 has an hour of 00, and the 12:xx hour holds both noon and
midnight: in a 1-in-10 sample of the starter, 41 of 61 frames stamped 12:xx are infrared, against 0 of 78 at 11, 13, 14
and 15. The camera most likely wrote 12 AM as 12. "Which nights" answers must use the frame's infrared state or treat
12:xx as ambiguous; this is the kind of wrong date the scenario wants. Frames also carry a burned-in strip with
temperature and moon phase, not the date.

**Alternates (metadata read):** Snapshot Serengeti season 1 has the empty share the scenario wants (82.4%) and 3-frame
bursts (394,200 of its 411,414 images), and its busiest camera (C03) holds 11,620 images in that season alone;
Wellington has few empties (16.9%) and minute-only times. Images of either were not downloaded.

**Gaps:** the camera triggers one frame per event here (every image is its own one-frame sequence), not the 3-frame
bursts many trail cameras make. The builder must write EXIF and file names (`IMG_0001.JPG`) from `date_captured`.

### I4 Home inspection or insurance

**Use:** `home-inspection-mobilehome` as the real thing and the needle source, `bd3-building-defects` for labelled
damage close-ups, `houses-dataset` as interior background. The mobile-home set is one person's phone walk through a
damaged home (Galaxy S24+ originals with EXIF, names like `20260110_142445.jpg`) with a JSON damage inventory of 14
items
by room, including water damage to the bathroom and hallway ceilings. The shipped JSON names no photos, so a person
must tag which of the 124 photos show each item (a short job). It has only 124 photos, so a 2k dump needs background.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [home-inspection-mobilehome](https://huggingface.co/datasets/beersrobert/mobile-home-interior-inspection) | 124 phone photos of one single-wide mobile home, 10 January 2026 (the README says 31, the JSON 79; the zip holds 124) | direct (HF) | 0.77 GB | `trailer_interior_documentation.json`: 4 rooms, 14 damage items by room with severity and cause, risks; no photo references (the README's per-item image list is not in the shipped file) | 124 images, 125 files, 0.78 GB; all of it |
| [bd3-building-defects](https://huggingface.co/datasets/chandrabhuma/building_defect_vqa) | BD3: 3,965 building-surface photos in 7 classes (algae, major and minor crack, peeling, plain, spalling, stain) | direct (HF) | 0.80 GB | one class per image | 793 images, 796 files, 0.32 GB; test split (793 images) with labels |
| [houses-dataset](https://github.com/emanhamed/Houses-dataset) | 535 listings x 4 photos (bathroom, bedroom, kitchen, front) | direct (GitHub) | ~180 MB | room type per photo | 2,140 images, 2,142 files, 0.19 GB; all of it |
| [RIUADA](https://zenodo.org/records/23536529) | facade defects after the 2024 Valencia floods: 215 photos, 1,159 expert-labelled crops (damp stain 158) | gated: Zenodo restricted, needs a request | | defect class per crop | none |
| crack sets on Mendeley Data and Zenodo (e.g. [BanglaCrack-10](https://data.mendeley.com/datasets/wd8zjmgx4b/1)) | crack and spalling photos | read only | | cracks only | rejected: no water damage or mould |

**Gaps:** no open set of real inspection or insurance-claim photo folders at 2k was found; Roboflow and Kaggle sets need
an account. Water damage and mould inside homes appear only in the mobile-home set (a few items) and as "stain" and
"algae" close-ups in BD3. "Every photo of water damage in the basement" needs a person to label the background, or the
needles kept to the mobile-home and BD3 photos.

### I5 Family archive scans

**Use:** `ia-family-photos-0487` as the background and `fgnet` for the query-by-example needles. The Internet Archive
item
is 1,376 scans of one family's prints from the 1950s-1960s; the uploader says many are undated and the years guessed,
and the EXIF holds the scanner (CanoScan LiDE 210) and the 2016 scan time, not the photo's date: the scenario's mess.
Checked by eye on 8: black-and-white and colour prints, some faded or overexposed, the same people at different ages.
FG-NET is 1,002 scanned photos of 82 people from age 0 to 69, with identity and age in each
file name, so "find this person" has a known answer across decades. Mixing them gives a dump whose "grandma" answers are
known; the family's own people would need a person to label.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [ia-family-photos-0487](https://archive.org/details/family-photos-0487) | 1,376 scanned family prints, 1950s-1960s, one family | direct (IA) | 0.83 GB | none (years guessed by the uploader) | 1,377 images, 1,379 files, 0.83 GB; all 1,376 scans, the IA thumbnail and the item metadata XML |
| [fgnet](https://huggingface.co/datasets/34data/human-fg-net-real-face) (unofficial mirror) | FG-NET: 1,002 scans, 82 people, ages 0-69; original FG-NET site reported gone | direct (HF) | 45 MB | identity + age in file name (`002A03` = person 2, age 3) | 1,002 images, 1,004 files, 91 MB; all 1,002 images |
| [Nobody's Album](https://archive.org/details/nobodys-album-found-120-envelope-6-1950s) (IA, 10 items) | found 120 negatives from one Portland estate, scanned 2026, grouped by envelope | direct (IA) | 0.70 GB | envelope-level date guesses | none (alternate background) |
| [CACD](https://bcsiriuschen.github.io/CARC/) | 163,446 web photos of 2,000 celebrities, 2004-2013 | direct (Google Drive, unverified) | 3.5 GB | identity + rough year | rejected: celebrity web photos over 10 years, not family scans |
| [DEW](https://service.tib.eu/ldmservice/dataset/rdr-doi-10-22000-44) | 1,029,710 Flickr photos 1930-1999 with year labels (paper record read) | blocked: the TIB page answers with a bot check | unknown | year per photo | none |

**Gaps:** no open 8k family archive with people labelled was found. The Flickr Commons route needs the Flickr API (an
account) and Flickr rate-limits direct image fetches.

### I6 Construction time-lapse

**Use:** `anu-physics-webcam`. It is a real building-site webcam in Canberra, one 4096x2160 frame an hour from
19 November 2018 to 29 September 2021 (22,832 files in four zips: 22,455 frames, plus 323 zero-byte frames and 54
Synology `@eaDir` sidecar files, which a real NAS export also has), with the time in each file name (written DD-MM-YYYY
until 14 June 2021, then YYYY-MM-DD). Checked by eye on
the noon and hourly frames: demolition in late 2018, then the tower crane. **Up:** a mobile crane arrives on 23 August
2019 about 10:00, the first mast sections stand at 15:00, the mast is up without its jib at 07:00 on 24 August, and the
jib is on by 14:00. **Down:** on Saturday 20 February 2021 a mobile crane is beside it from 09:00, the jib is off by
13:00, the mast shrinks through the afternoon and is gone at 17:00 (last seen 16:00); the mobile crane is gone at 18:00.
A second, distant crane stays at the top right edge into April 2021, a natural decoy, and a hot-air balloon crosses at
07:00 on 21 February 2021. So "when was the crane taken down?" has an hour-level answer, read by eye from downloaded
frames.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [anu-physics-webcam](https://archive.org/details/anu-physics-building-webcam-archive) | hourly 4K frames of one building site, Nov 2018 - Sep 2021, gaps in Nov 2019 and Nov 2020 | direct (IA; ~0.5 MB/s per connection) | 15.8 GB | capture time per frame (file name); events by eye only | 1,918 frames (9 of them zero-byte, as in the source) and 2 `@eaDir` sidecar files, 1.47 GB; noon frame of every 2nd day + 1st of each month, and every hourly frame of February 2021 (crane removal) and August 2019 (crane erection) |
| [purdue-webcams](https://engineering.purdue.edu/ECN/WebCam) | ~25 archived Purdue construction cameras, 4K with EXIF; e.g. cam87 lists 1,311 frames 2024-03 to 2026-11 | direct (one frame fetched) | ~1.7 GB for cam87 (estimate) | time in name and EXIF | none: past months keep one frame a day |
| [MTU webcam mirror](https://mirrors.lug.mtu.edu/mtuwebcams/) | East Hall construction, 5-minute frames (per search results) | blocked: folder empty, README 404 | | | none |
| AMOS ([site](http://amos.cse.wustl.edu/)) | archive of many outdoor webcams | blocked: no response | | | none |
| construction time-lapse videos on IA (e.g. [Broadway Bridge Replacement](https://archive.org/details/coewa-Broadway_Bridge_Replacement_-_February_2015_to_March_2016)) | edited time-lapse videos | read only | | | rejected: edited video, frame times unknown |

**Gaps:** no 5-minute archive was found open; ANU is hourly, so 50k frames at 5 minutes would need hourly frames from
several cameras or a shorter span. The crane times above were read by me from the frames, not from a label; a person
should confirm them (hourly frames of August 2019 and February 2021 are downloaded). A University of Melbourne
construction time-lapse site
(`secsys-timelapse.unimelb.edu.au`) failed TLS verification and was not opened.

## 5.3 Long audio, one file

### A1 Home while away

**Use:** `sins`, Node 7 (living room). SINS is one real home recorded without a break for a week, so one node's
one-minute files for one day join into the single 24-hour file the scenario needs, and its activity labels give the
answers at activity level. 2 February 2017 fits "while away": the occupant is out 08:29-14:59 and 15:11-17:30, a visit
is labelled 14:59-15:08, the TV is on 18:43-21:56 and the vacuum cleaner runs 22:04-22:12. SINS has no smoke alarm,
doorbell or dog, so those needles come from FSD50K (see M4), mixed in at a set SNR (DESIGN 7.1, way 2);
the real TV hours give the decoys. The whole day is downloaded (1,441 files, no file missing).

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [sins](https://github.com/KULeuvenADVISE/SINS_database) Node 7 ([Zenodo 2547315](https://zenodo.org/records/2547315)) | A holiday home, one occupant, 30 Jan 04:38 to 5 Feb 09:32, 2017 (6.2 days); 12 nodes of 4 microphones in 5 rooms; Node 7 is on the living-room wall facing the sofa, dining table and kitchen. Checked: 16 kHz, 16-bit, 4-channel WAV in ~60 s files (Node 7: 59.9 s), named by the Pi's clock to the ms; no gap over 2.1 s between files all week | direct (Zenodo zips; range requests work) | 46.25 GB for Node 7's week; ~539 GB for all nodes | `labels.csv`: 644 spans for the week, ms wall-clock start and stop, 21 activity classes plus room presence (absence, visit, watching tv, vacuumcleaner, cooking, dishwashing, eating, calling, working, ...). Activity level only: no sound-event labels | all of 2 February 2017 with no missing file: 1,441 one-minute files, 23.98 h (23:59:25 on 1 Feb to 00:00:15 on 3 Feb), lossless 4-ch FLAC, 3.25 GB (11.05 GB as WAV); all 6 label files, README, floor plan, licence |
| [dcase2018-task5](https://zenodo.org/records/1247090) | The same SINS audio cut into 10-s segments for DCASE 2018 | read only (search result) | ~46 GB (unverified) | activity per segment | rejected: cut into clips; the continuous source is better |
| [DEMAND](https://zenodo.org/records/1227121) | 16-channel noise recordings incl. a living room (DLIVING) and kitchen | direct (record checked) | 15 recordings; DLIVING 16 kHz zip ~100 MB | none | rejected: a few minutes per room (estimated from zip size) |
| [24 Hour Austin Rich Field Recording](https://archive.org/details/24-hour-austin-rich-field-recording) (Internet Archive) | one 24-hour MP3 (86,400 s, 1.38 GB) | direct (metadata checked) | 1,382,401,788 bytes | none | not downloaded: no description; a person must listen to know whether it is indoors |

**Cheapest way to one day:** each node's week is 9-10 zips of ~5 GB with the one-minute WAVs in random time order, so a
day touches every zip. Reading each zip's central directory by range (3-4 requests of at most ~0.5 MB per zip) gives
every member's name, offset and size; one day of Node 7 is then 1,441 range requests totalling 4.89 GB (23.98 h of
audio), against 46 GB for the node. `sins_fetch.py` does this and checks each member's CRC32.

**Gaps:**
- No second real home recording of hours or days was found that downloads without an account. Searches of Zenodo, the
  Internet Archive and the web found only SINS, its DCASE cut, and minutes-long noise sets.
- SINS labels are activities (and whole-home: the occupant's room, not what Node 7 hears). Questions about sounds need
  mixed needles; "when did the TV come on" can use the labels directly.
- Between two consecutive SINS files the clock leaves a median ~0.08 s unaccounted for (max 2.1 s for Node 7). I did not
  settle whether samples are lost there. A 24-hour file built by joining files will have ~1,440 such seams; the shortcut
  probe should look at them.
- Measured: every SINS channel carries a DC offset of about half of full scale (mean 0.49-0.53 on a ±1 scale), so the
  useful signal is small (living room at night: AC RMS about -38 to -50 dBFS; bedroom at night about -45 to -51 dBFS).
  Remove the DC before mixing needles or setting an SNR, and decide whether the dump keeps it (it is "real mess", but a
  tool may stumble on it).
- Not downloaded: the sync `.mat` files (one per WAV, ~1.6 MB each), which align nodes to ±0.5 s.

### A2 Overnight recording

**Use:** `sins`, Node 10 (bedroom), the night of 1 February 2017, 01:30-10:15. It is a real bedroom with a real sleeper:
the labels say `sleeping` 01:40:04-10:07:28, with `bedroom` and `dressing` around it, and the file is mostly quiet room
noise, as the scenario wants; that window is downloaded whole. There is no baby and no labelled sleep talk, so those
needles are mixed in: real infant cries from `donateacry` (phone recordings, not FSD50K) or FSD50K's crying classes
(see M4), and sleep talk from Dan or a scripted recording. `apsaa` gives more real whole nights (a
hospital sleep room, with snoring and apnea events labelled) if a second night or a snoring question is wanted.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [sins](https://github.com/KULeuvenADVISE/SINS_database) Node 10 ([Zenodo 2555137](https://zenodo.org/records/2555137)) | Bedroom node of the SINS home; 7 labelled sleeping spans of 3.0-8.5 h. Checked: 4-channel 16 kHz WAV, 61.5 s files, no gap over 1.3 s all week | direct | 43.58 GB for the node's week | `sleeping`, `bedroom`, `dressing` spans (ms). No sleep talk, snore or cry labels | the night of 1 February 2017, 01:29-10:15 with no missing file: 512 one-minute files, 8.75 h, lossless 4-ch FLAC, 1.18 GB (4.03 GB as WAV); labels as in A1 |
| [apsaa](https://zenodo.org/records/14096541) | 32 full nights in the sleep unit of a hospital in Jaén (2021-22): one room-audio WAV per subject plus polygraph CSVs | direct (zip listed by range; one subject fetched) | 3,927,427,834 bytes | per-subject polygraph events (apneas, hypopneas, desaturations) with times; a snore-envelope signal | 1 subject (20210928B): 7.88 h WAV, mono 4 kHz 16-bit (226,948,750 bytes), its annotations (201 snore, 112 hypopnea, 164 desaturation, 13 apnea events (11 obstructive, 1 central, 1 "no effort"); start HH:MM:SS from the start, duration to 0.01 s) and 7 polygraph CSVs |
| [psg-audio](https://doi.org/10.11922/sciencedb.00345) | 212 sleep-lab nights with an ambient microphone 1 m above the bed (per the [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8333307/)); audio inside 1-hour EDF files | blocked: the ScienceDB page is a JavaScript app and I found no scriptable file URL; not logged in | unknown | per paper: snore, apnea, arousal, sleep-stage annotations | none; needs a person to find a download URL |
| [donateacry](https://github.com/gveres/donateacry-corpus) | Infant cries recorded by parents with a phone app; needles for "when did the baby cry" | direct (GitHub API checked; repo zip downloaded) | 70,807,712 bytes (GitHub zip) | cry reason per clip (hungry, tired, discomfort, belly pain, burping) | whole repo (97 MB): 457 cleaned WAV clips, 8 kHz mono, ~7 s each (53 min); 712 raw Android .3gp (AMR-NB 8 kHz) and 416 raw iOS .caf (16 kHz) uploads (129 min) |
| [ICSD](https://huggingface.co/datasets/QingyuLiu1/ICSD), [CryCeleb2023](https://huggingface.co/datasets/Ubenwa/CryCeleb2023), [DreamCatcher](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher) | infant cry and snoring sets; DreamCatcher is 420 h of sleep from earbud microphones with sleep-talk labels (read only) | gated: Hugging Face request form or accept-terms gate | ~4 h (ICSD); 420 h (DreamCatcher), per their pages | ICSD onset/offset; DreamCatcher somniloquy events | not downloaded (gated) |

**Gaps:**
- No open real sleep-talk recording was found. The one labelled set (DreamCatcher) is gated and uses in-ear microphones.
  Internet Archive title searches for "sleep talk" and "somniloquy" returned songs, poems and podcasts. Sleep talk needs
  Dan's own nights or a scripted recording.
- No real baby-monitor night was found. Cries must be mixed into an adult's bedroom (SINS) or a clinic room (APSAA).
- APSAA's audio is 4 kHz mono (checked on one subject), so it is phone-call bandwidth; the microphone position is not
  stated in the record. The clinic room is not a home bedroom. Its record restricts use to non-commercial research and
  forbids redistribution.
- PSG-Audio is the best-labelled overnight set but could not be fetched by script; unverified.

### A3 Meeting or interview (audio)

**Use:** `ami`. The ES2002a-d series (4 meetings, 2.39 h) has audio and video; the ES2016a-d series (2.12 h, audio only
here)
is better for "when was it decided", since all four of its meetings have decision spans. `icsi` adds real, unscripted
research meetings of about an hour each. All decision, topic, dialogue-act and agreement annotations for the corpus come
in one 22.9 MB zip.

How "when was it decided or agreed" is answered from AMI's files (NXT XML in `ami_public_manual_1.6.2.zip`):
- `words/<m>.<A-D>.words.xml`: every word with `starttime` and `endtime` in seconds, 10 ms resolution.
  The words are human transcripts; their times come from forced alignment (`00README_MANUAL.txt`). All other layers
  point at word ranges, so every label below resolves to a time span this way.
- `decision/manual/<m>.decision.xml`: decision spans (47 meetings; ES2002a has 3, ES2002d 8, ES2002b and c none). Series
  with all four meetings annotated: ES2015, ES2016, IS1004, IS1006, IS1008, TS3005.
- `abstractive/<m>.abssumm.xml`: abstract, actions, decisions and problems sentences; `extractive/<m>.summlink.xml`
  links
  each sentence to the dialogue acts it summarises, so a summary's decision sentence maps to times.
- `dialogueActs/<m>.<A-D>.dialog-act.xml`: acts with 15 types (Inform, Suggest, Assess, Offer, Elicit-*, Be-Positive,
  Be-Negative, Backchannel...); `dialogueActs/<m>.adjacency-pairs.xml` links acts with POS (support), NEG (objection),
  PART (partial agreement), UNC and ELA. "When did B agree to the price" is a POS pair whose first act mentions the
  price.
- `topics/<m>.topic.xml`: topic segments with a scenario topic type (7-17 per ES2002 meeting).
- Speakers are A-D per meeting; `corpusResources/meetings.xml` maps them to roles and channels.

ICSI (`ICSI_plus_NXT.zip`) has human topic segmentation with free-text descriptions, hot spots with start and end times
(0.1 s), extractive and abstractive summaries, dialogue acts and words.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [ami](https://groups.inf.ed.ac.uk/ami/download/) | as above | direct | ~100 h; Mix-Headset alone ~11 GB for all meetings (estimate: 40-85 MB each) | as above | ES2002a-d and ES2016a-d Mix-Headset (4.5 h); Array1-01 far mic for ES2002a-d; all headsets for ES2002a; both annotation zips |
| [icsi](https://groups.inf.ed.ac.uk/ami/icsi/download/) | ICSI Meeting Corpus, 75 real meetings, ~72 h | direct | 7.87 GB headset mixes (listing); 53 MB annotations | as above | annotation zips; Bmr005 (71 min) and Bmr019 (60 min) headset mixes |
| [diarizers-community/ami](https://huggingface.co/datasets/diarizers-community/ami) | AMI full-meeting audio (ihm and sdm) with speaker turns, parquet | direct (not gated) | 25 shards of 0.3-0.55 GB per config | speaker turns | not downloaded; a faster mirror than Edinburgh |

**Gaps:** No long interview-tape collection with timed transcripts was verified. Leads, unverified: oral histories
indexed
with OHMS (time-coded segments) at Hagley (Brandywine Valley, 200+ interviews) and the Yiddish Book Center (600+
timecoded
indices); whether their audio and indices download without an account is unknown. AMI and ICSI meetings run 20-70
minutes, so a 3-hour or 6-hour dump is a folder of consecutive meetings, which is what a real meeting folder looks like.
AMI's scenario meetings are role-played; ICSI's are real.

### A4 Field recorder

**Use:** `birdvox-full-season` for the dump and the annotated soundscape sets for the truth. No single open set is "3
nights outdoors with labelled owls and frogs". The closest real shape is BirdVox: nine recorders that ran dusk to dawn
every night from 3 August to 8 December 2015 near Ithaca, NY. Three consecutive nights of one unit (unit03, 22-24
September: 832 + 869 + 947 MB) make the dump; the middle night is `birdvox-full-night` unit03 (the zip member is
868,993,719 bytes, the same size as that file), whose 9,113 flight calls are timed. All three nights are downloaded
(32.8 h, 2.65 GB), so the A4 dump can be built now. That supports "when did the migrants start calling" and "find every
thrush call" by labels. Owl and frog questions need needles mixed in: owl calls cut from labelled boxes in
`soundscapes-westernghats` or `soundscapes-amazon`, and frog choruses from `anuraset` minutes. `soundscapes-westus`, the
named source, is 33 labelled hours at dawn and dusk, not nights; it suits species questions and M5.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [birdvox-full-season](https://zenodo.org/records/5791744), with [birdvox-296h](https://zenodo.org/records/5856260) | 9 ROBIN recorders, nightly dusk-dawn FLACs (24 kHz mono), Aug-Dec 2015; unit03.zip lists 82 nights (checked by range). BirdVox-296h is 148 two-hour WAV pieces of other nights of the same season | direct | 624,097,770,622 bytes (unit03 72.7 GB); 296h: 36.2 GB | none of its own; one night equals BirdVox-full-night. 296h: 26,138 point events with species/family | season: unit03's nights of 22 and 24 Sept, one FLAC each (10.88 h, 832,323,263 bytes; 10.97 h, 947,420,387 bytes; zip members by range, CRC32 checked). With the full-night file below this is three consecutive nights of one recorder, 32.8 h. 296h: annotations, taxonomy, GPS |
| [birdvox-full-night](https://zenodo.org/records/1205569) | the same recorders, night of 23 Sept 2015, 6 units × ~10 h, continuous | direct | 5,645,204,966 bytes audio | 35,402 flight calls as points (s, Hz), no species | unit03 whole night: 10.93 h FLAC, 868,993,719 bytes; all 6 units' call CSVs, README, start times, GPS |
| [soundscapes-westus](https://zenodo.org/records/7050014) | 33 one-hour SWIFT recordings, Sierra Nevada, May 2018, hours 04-09 and 17-19 | direct | 3,249,297,967 bytes | 20,147 boxes (start/end to 0.1 s, Hz band, species), 56 species; no owls, no frogs | 2 hours (SNE_001 and SNE_002, 9 May 2018 05:00-07:00; 252 + 591 boxes), FLAC 32 kHz mono, 0.19 GB; annotations, species, description, location |
| [soundscapes-westernghats](https://zenodo.org/records/16881041), [-amazon](https://zenodo.org/records/7079124), [-neus](https://zenodo.org/records/7079380), [-hawaii](https://zenodo.org/records/7078499), [-sierrasouth](https://zenodo.org/records/7525805), [-coffee](https://zenodo.org/records/7525349) | the rest of the same family (Cornell), each with `annotations.csv` | direct | 9.5 (+40.3 unlabelled), 2.4, 27.8, 5.8, 1.4, 3.8 GB | boxes per species. Owls: Western Ghats 180 owl boxes (Indian Scops-Owl 147) in evening files; Amazon 187 (pygmy-owls, screech-owl); NE US 20 Barred Owl; coffee 1 screech-owl + 42 potoo | all six annotation, species and description files; no audio |
| [anuraset](https://zenodo.org/records/8342596) | 1,612 one-minute recordings of frog choruses at 4 Brazilian sites, on a schedule | direct | 7,213,646,303 bytes raw | per-minute strong labels: onset/offset, species, call quality | site INCT41, 2020-02-13: all 23 one-minute files of that date (00:15-23:30, WAV 22,050 Hz 2-ch); species, weak and strong labels for all 1,612 files. The folder also holds site INCT17's files, fetched for M5 |

**Gaps:**
- No continuous multi-night recording with owl or frog labels was found. Web searches for annotated owl and frog sets
  found per-clip or per-interval labels (Rana sierrae 10-s clips on Dryad, a spotted-owl dawn set on Zenodo with 2-s
  intervals), not continuous nights.
- The NE US set has night hours with Barred Owl, but its description says each hour came from a randomly chosen one of
  30 recorders, so a day of it is not one recorder's night.
- BirdVox's upward-facing horn microphones reject sound from the side; owls and frogs would be faint even if present. No
  one has labelled them.
- Western US hours on one date may come from different sites (the description says sites spanned a gradient).

### A5 Car or machine

**Use:** `mimii` (slider, machine id_06, 6 dB) for the needles, and no long base yet. MIMII is real factory machine
sound with normal and abnormal clips, but every file is a 10-s clip, so a 2-hour or 24-hour file would be clips joined
end to end, which DESIGN 11.2 warns against. I checked whether clips numbered in a row are pieces of one recording: the
level step across the join of clip n and n+1 (median 1.8 dB over 50 ms, abnormal id_06) is smaller than between random
clips (3.6 dB) but larger than between neighbouring windows inside one clip (1.1 dB); for normal clips 1.9, 2.6 and 1.0
dB. So the numbering seems to follow time, but the clips do not look contiguous; joining them leaves seams. (The same
test on SINS files gives 0.6 vs 0.5 dB, as expected of a continuous recording.) No open long continuous machine or
in-car recording with a known fault time was found. The honest options are: (1) Dan records a long drive or a running
appliance and a fault is made at a known time (DESIGN 7.1, way 4); (2) a long real background with MIMII abnormal clips
mixed in at a set SNR, once a long background is found.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [mimii](https://zenodo.org/records/3384388) | fan, pump, slider, valve; slider has 4 machine ids; 12 zips by SNR (-6, 0, 6 dB). Checked: 8-channel 16 kHz WAV, 10 s; slider id_06 at 6 dB has 534 normal and 89 abnormal clips | direct | 100,243,834,344 bytes | normal/abnormal per clip (folder); no onset inside clips | slider id_06, 6 dB: all 89 abnormal and 111 of 534 normal clips (0.51 GB, 0.56 h) |
| [mimii-due](https://zenodo.org/records/4740355), [mimii-dg](https://zenodo.org/records/6529888) | later MIMII sets (adds gearbox, bearing; domain shift) | direct (records checked) | 9.4 GB; 4.4 GB | normal/anomaly per clip | none: same clip shape |
| [toyadmos](https://zenodo.org/records/3351307), [toyadmos2](https://zenodo.org/records/4580270) | toy car, conveyor, train; per a search snippet of its paper, ToyADMOS includes 10-minute continuous recordings. ToyADMOS2 adds recorded factory-like noise | direct (records checked); ToyADMOS is split 7z volumes; ToyADMOS2's env-noise zip listed by range | 197.1 GB; 169.4 GB | normal/anomalous per file | none: one toy is 58-71 GB of 7z volumes that cannot be range-read; ToyADMOS2's noise is 36,000 short MP4 clips |
| [cavemove](https://zenodo.org/records/17404917); Internet Archive in-car ([aporee 70063](https://archive.org/details/aporee_70063_81610), [forest road](https://archive.org/details/CarInteriorAmbienceDrivingUpAForestServiceRoad)) | real cabin noise in 4 cars by speed and window opening, 8-channel; a real binaural drive (582 s) and a car interior (450 s) | direct (cavemove zip listed by range; IA metadata checked) | 7.6 GB; 169 MB; 71 MB | speed and window per file; none | rejected as the base: ~2 min per condition (from file sizes) and minutes, not hours; could give cabin backgrounds |
| [WSU drill run-to-failure](https://researchers.westernsydney.edu.au/en/datasets/ultrasonic-acoustic-dataset-of-industrial-drill-degradation-and-f/) (DOI 10.26183/v7kh-b648) | ultrasonic MEMS microphones in a CNC machining centre; 17 drills each run until critical failure (18-107 holes); WAV per drill (read only) | blocked: the [data page](https://research-data.westernsydney.edu.au/published/eb7b78503f7f11f1a4923741fec30ff7) and the file listing returned HTTP 403 (CDN "Access Denied") to a script | unknown | holes to failure per drill; no failure timestamps stated | none; a person should try it in a browser. The only run-to-failure audio found |

**Gaps:**
- No 2-hour real drive and no 24-hour real factory recording that downloads without an account was found (Zenodo and
  Internet Archive searches; title search only on IA).
- No machine recording where a fault starts at a known time inside one file was found. Two might be close, both
  unverified: ToyADMOS's 10-minute continuous recordings (too large to sample), and the WSU drill set (blocked by its
  CDN; a drill is recorded until it fails, but the page gives hole counts, not times).

## 5.4 Many short audio files, hours in total

### M1 Voice memos

**Use:** `ia-voice-memos` for the dump's shape, `voicenotes-rosehill` for findable content, `mtg-qbh` for the hummed
melodies. `ia-voice-memos` is one uploader's phone-recorder folder (121 M4A files, named `20180121 100152.m4a`,
January 2018 to December 2019), but nothing says what is in it, and nobody has listened to it yet. `voicenotes-rosehill`
is about 110 real voice notes from one
person with app transcripts. Pool them, rename all to one recorder's pattern, and plant 2-5 MTG-QBH hums.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [ia-voice-memos](https://archive.org/details/20190504-171211) | "2018-2019 voice memos": 121 AAC M4A from a phone recorder, 1.7 s to 63 min (median 2.3 min), 2018-01-21 to 2019-12-30 | direct (checked; MD5s match IA) | 304 MB | none: dates from names only | 121 media files (10.0 h), 122 files in all, 0.30 GB |
| [voicenotes-rosehill](https://huggingface.co/datasets/danielrosehill/Voice-Note-Audio) | Daniel Rosehill's notes from the voicenotes.com app: 101 + 7 MP3 notes (15 s to 7 min), and 19 scripted notes read aloud, each also as a denoised copy (38 WAVs); also [Noisy-Voice-Notes](https://huggingface.co/datasets/danielrosehill/Noisy-Voice-Notes) and [STT-Voice-Notes-Evals](https://huggingface.co/datasets/danielrosehill/STT-Voice-Notes-Evals) | direct (checked) | ~0.46 GB | ASR transcripts (ElevenLabs, AssemblyAI; uncorrected), note type, time of day; exact text for the scripted ones | 146 media files (4.2 h), 473 files in all, 0.75 GB |
| [mtg-qbh](https://zenodo.org/records/1290712) | 118 hummed or sung melodies, 17 people, laptop microphone, 11-98 s | direct (checked) | 223 MB | the song each one is | 118 media files (0.9 h), 122 files in all, 0.28 GB |
| [HumTrans](https://huggingface.co/datasets/dadinghh2/HumTrans), [CHAD hummings](https://huggingface.co/datasets/amanteur/CHAD_hummings) | 56 h and 5,314 hummings | direct (API checked) | 14.7 GB, 1.6 GB | MIDI; song group | no: MTG-QBH is enough |
| IA "Voice Memos 1-4", "5", "My Voice Memos Collection!" | iPhone memo exports (502, 104, 57 M4A) | direct (listing checked) | 2.8, 1.4, 0.7 GB | none | no: titles and lengths say they are mostly radio recorded on a phone |

**Gaps:** no open set of real memos with human transcripts. `voicenotes-rosehill`'s `recording_date` is 2025-10 for all
101 notes, which looks like the export date, not the recording date. I did not listen to `ia-voice-memos`, so its
content (speech, music or radio) is unknown. A 400-memo dump needs Dan's own memos or the scripted set.

### M2 Messaging voice notes

**Use:** `whatsapp-ptbr-bernardoai` plus `voicenotes-rosehill`, if a small M2 is wanted now. Re-encode to Opus with
WhatsApp-style names (`PTT-20250612-WA0007.opus`). The real fit, EMOVOME (999 WhatsApp voice messages with human
transcripts), needs a signed agreement.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [EMOVOME](https://zenodo.org/records/10694370) | 999 real WhatsApp voice messages, 100 Spanish speakers | gated: signed agreement for the audio (checked: the open zip's `Audios/` folders are empty) | 1.3 GB zip (mostly models) | human transcripts, emotion labels | no |
| [whatsapp-ptbr-bernardoai](https://huggingface.co/datasets/BernardoAI/wpp_pav_transcrito_openai) | 198 WhatsApp audio clips, Brazilian Portuguese, Nov 2020 to Mar 2021, converted to 16 kHz WAV; 0.7-18.5 s, so probably segments of messages | direct (checked) | 52 MB | `original_text`, verbatim-looking (keeps fillers; author undocumented), and a GPT-4o transcript | 198 media files (0.5 h), 202 files in all, 0.10 GB |
| [voicenotes-rosehill](https://huggingface.co/datasets/danielrosehill/Noisy-Voice-Notes) | see M1; app voice notes, noisy places | direct (checked) | ~0.46 GB | ASR transcripts | (counted under M1) |
| [ivrit-ai crowd-whatsapp](https://huggingface.co/datasets/ivrit-ai/crowd-whatsapp-yi) | crowd-sourced WhatsApp recordings | gated: HF agreement | unknown | transcripts (by name) | no |

**Gaps:** no open source near 1,000 real voice notes; the open ones were converted from Opus, so the real codec is lost.
Mozilla's Common Voice Spontaneous Speech is behind a Mozilla Data Collective account (read, not checked).

### M3 Voicemails

**Use:** `ia-industry-voicemail` as the dump and `robocall-ftc` for questions with text truth. `ia-industry-voicemail`
is real business voicemail, one message per file, released in opioid and JUUL litigation (UCSF Industry Documents
Library on Internet Archive; about 1,650 voicemail-type items among 3,073). `robocall-ftc` adds 1,427 real robocall
messages with Whisper transcripts and the FTC case each belongs to, which makes "the voicemail about the Amazon order"
answerable. `ia-markpines-answering` is real home answering-machine tape, 1984-2001.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [ia-industry-voicemail](https://archive.org/details/industry-archives-audio) | real voicemails (titles `VOICEATT.WAV`, `MSG0001.WAV`, `VoiceMessage.wav`) from discovery in opioid and tobacco cases, one per item; probe of 256: WAV GSM 6.10 8 kHz (126), WAV G.711 mu-law 8 kHz (45), M4A AAC (74), MP3 (11); 4 s to 4.7 min, median 61 s | direct (checked; sizes and MD5s match IA) | 3,073 items, ~1,650 voicemail-type (estimate from titles) | none (title, date, case subjects) | 256 media files (5.7 h), 512 files in all, 0.12 GB: 256 of a 300-item selection, stopped at a time limit |
| [robocall-ftc](https://github.com/wspr-ncsu/robocall-audio-dataset) | 1,427 FTC robocall recordings (1,432 rows in its metadata), 16 kHz WAV, median 22 s | direct (checked) | ~1.15 GB unpacked | Whisper-medium transcript (model), FTC case per call | 1427 media files (10.1 h), 1461 files in all, 1.15 GB |
| [ia-markpines-answering](https://archive.org/details/Mark_Pines_Answering_Machine_Messages) | 46 digitised answering-machine tapes of one household, 1984-2001 | direct (checked) | 38.7 GB | IA Whisper transcripts with times (model); dates in names | 3 media files (0.5 h), 94 files in all, 0.35 GB: 3 tapes and all 45 transcripts |
| [minds14](https://huggingface.co/datasets/PolyAI/minds14) en-US | 563 crowd-sourced phone recordings of bank requests, mu-law 8 kHz | direct (checked) | 3.9 GB, all languages | intent per call; transcripts with ASR-like errors | 563 media files (1.3 h), 566 files in all, 73 MB |
| [LDC Voicemail Corpus I, II](https://catalog.ldc.upenn.edu/LDC98S77) | ~3,900 real voicemails with transcripts | gated: LDC licence (read) | | transcripts | no |

**Gaps:** the real voicemails have no transcripts; a person must transcribe or check ASR for each message a question
uses. The robocall audio was resampled to 16 kHz WAV, so its codec is not real. Mark Pines' tapes hold many messages
each and need cutting at the beeps. None of the sources has "the dentist about rescheduling" by design: questions must
come from what the messages say.

### M4 Sample library

**Use:** `fsd50k`. The eval split alone is 10,231 WAVs, the dump's size, with human-verified labels in 200 classes; the
dev clips add more needles. Rename the files away from Freesound ids (a leak) and put them in a sample-pack folder
layout. "Every snare with reverb" has no label: FSD50K labels say "Snare_drum", not effects.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [fsd50k](https://zenodo.org/records/4060432) via [Fhrozen/FSD50k](https://huggingface.co/datasets/Fhrozen/FSD50k) | 51,197 Freesound clips, WAV 44.1 kHz mono, 0.3-30 s | direct (checked; mirror verified, below) | 34.5 GB | clip-level labels, 200 classes; eval labels exhaustive | 13803 media files (35.4 h), 13825 files in all, 11.29 GB |
| [esc50](https://github.com/karolpiczak/ESC-50) | 2,000 five-second clips, 50 classes | direct (checked) | 0.65 GB zip | one class per clip | 2000 media files (2.8 h), 2004 files in all, 0.88 GB |
| Freesound packs | the original uploads | gated: account and API token (checked: HTTP 401) | | tags | no |
| [Sonniss GDC bundles](https://sonniss.com/gameaudiogdc) | professional SFX library, ~207 GB over 8 years | blocked: HTTP 403 to scripts (checked); torrent not allowed | | file names | no |
| CLAPv2/FSD50K, philgzl/fsd50k mirrors | FSD50K re-encoded | direct | | | no: resampled to 48 kHz (checked one shard) or Opus (per README) |

**How the mirror was verified:** all 51,197 file names and byte sizes on the HF mirror equal the entries of Zenodo's
`FSD50K.eval_audio.zip` and `FSD50K.dev_audio.zip` central directories, read by HTTP range requests. All 10,231 eval
clips I downloaded match Zenodo's CRC32 and size (`fsd50k/_verify/`). The labels and metadata files are byte-identical
to Zenodo's `FSD50K.ground_truth.zip` and `FSD50K.metadata.zip`, whose MD5s match the Zenodo record.

**Starter:** all labels and metadata, the whole eval split, and 3,572 dev clips from 30 home and camera classes (alarm,
doorbell, knock, door, slam, glass, shatter, bark, crying, screaming, gunshot, siren, telephone, ringtone, footsteps,
microwave, toilet flush, tap, and others; at most 150 per class, seeded random; list in `fsd50k/_selection/`). These
are the needles A1, A2 and M6 mix in.

**Gaps:** dev clips outside those 30 classes (about 37,400) are not downloaded; `fetch_full` in the jsonl gets them.

### M5 Bat or bird recorder on a schedule

**Use:** `anuraset`. Its files are the real one-minute files of SM4 recorders set to one minute every 15 minutes,
named by site, date and time, with per-file presence labels for 42 frog species and timed call labels. Questions become
"which recordings have [species] calling?". It is a stratified random sample of night files (1,612 of the full set),
not a whole SD card, and frogs, not owls.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [anuraset](https://zenodo.org/records/8342596) | 1,612 one-minute WAVs from 4 Brazilian sites, 1-in-15-min schedule | direct (checked; each file CRC32-checked) | 7.2 GB | per file: activity 0-3 per species; per call: start and end | 160 media files (2.7 h), 1776 files in all, 0.85 GB: the first 160 of site INCT17's 354 files (2019-11-13 to 2020-02-08), all labels; with A4's 23 INCT41 files, all in the folder's `MANIFEST.jsonl` |
| [Dja AudioMoth, Cameroon](https://datadryad.org/dataset/doi:10.5061/dryad.866t1g1rh) | 38,065 AudioMoth files, first minute of every hour, 18 sensors, Feb-May 2020 | blocked for scripts: the API needs a token (checked: 401) and the web download shows an AWS WAF human check (403) | 176 GB | species present per file for 20,485 files, by local experts | no |
| [Dawn chorus, spotted-owl programme](https://zenodo.org/records/10252138) | 1,575 random 5-min SM4 samples at dawn | direct (checked) | ~19.8 GB | 141 files fully labelled at 2 s | no: 5-min samples; owls almost absent from its labels |
| [WABAD](https://zenodo.org/records/17293588) | annotated minutes from 70 sites, ~30 with AudioMoths | direct (checked) | 21.3 GB | every call to species with times | no: chosen minutes, so almost every file has birds |
| Kahl soundscape family (Hawaii, Sierra Nevada, ...) | long or multi-minute annotated soundscapes | see A4 | | | no |

**Gaps:** the best fit, Dja (real AudioMoth schedule, per-file presence, owls possible), needs an API token for scripts,
and its web download shows an "are you human" check; a person in a browser can probably fetch it without an account. I
found no open owl or bat set on a recorder schedule with labels. 160 files (183 with A4's) is under the 300-1,000 the
dump
wants; the rest of `raw_data.zip` needs more time on Zenodo.

### M6 Motion clips with sound

**Use:** `xd-violence` for now; `mavd-traffic` as a small real street camera with timed sound labels. XD-Violence
videos carry their own sound and its test set has frame spans for explosions, crashes, riots, fights and shooting.
"Which clips have an explosion?" or "...a car crash?" can be graded from those spans. Cut to 30-s clips at the span
edges; MP4 names need replacing. No source found answers "glass breaking" from its own labels; that still needs FSD50K
glass clips mixed into real camera sound.

| Source | What it is | Access | Full size | Truth | Starter downloaded |
| --- | --- | --- | --- | --- | --- |
| [xd-violence](https://huggingface.co/datasets/jherng/xd-violence) | 4,750 videos with sound from films and YouTube (CCTV, dashcam, phone, news), re-encoded by the dataset (mostly 640x360 H.264, AAC) | direct (checked) | 85.5 GB | class per video; frame spans for 800 test videos | 221 media files (7.5 h), 225 files in all, 2.76 GB |
| [mavd-traffic](https://zenodo.org/records/3338727) | street recordings in Montevideo: FLAC 44.1 kHz plus AVI video published at 176x144 with its own sound | direct (checked; MD5s match Zenodo) | 1.3 GB | timed traffic sound events (vehicle, part, action) | 14 media files (1.1 h), 62 files in all, 0.19 GB |
| [EPIC-KITCHENS + EPIC-SOUNDS](https://epic-kitchens.github.io/) | head-camera kitchen video with sound | direct (HEAD checked) | ~700 GB (estimate) | 78K timed sound events | no: see Kitchen under 5.1; 1-6 GB per file |
| [Dashcam Archive, Bristol VA](https://archive.org/details/daily-dashcam-2026-09-09-bristol-va) | daily raw dashcam MOV chapters | direct (checked) | 1.37 GB for 2026-09-09; days vary (V2) | none | no: probed one file, no audio track |
| [TAU Urban Audio-Visual 2021](https://zenodo.org/records/4477542), [VGGSound](https://huggingface.co/datasets/Loie/VGGSound) | 10-s real clips with sound | direct (listings checked) | 108 GB, 338 GB | scene class; one class per clip | no: no event times, and only in 4-17 GB archives |

**Gaps:** no open doorbell or security clips with sound and event labels. XD-Violence is partly edited material; the
YouTube test videos I took still include news and compilations, so the probe must check splices. Of the 221 starter
videos, 173 have frame spans (611 spans), 40 are labelled normal, 8 violent ones have no spans. MEVA's clips have no
sound (see V1).

## 5.5 Mixed dumps, and the large-file track

### X1 Scripted real day, X2 Whole phone dump

Both are Dan's own media and stay private, so there is nothing to find online. ATM-Bench (I1) also has 533 videos of the
same person in `data/raw_memory/video/`, not downloaded. With its photos they would make a public, mixed stand-in for
X2.

### L Large-file track

Built by re-muxing other scenarios' media. Real big single files are now on disk:
- a 6.12 GB 4K VP9 concert (`yt-concert-deltavista-band-2026`, V3);
- a 5.53 GB 11.6-hour feeder day (`feedercam-birdsuntold`, V6);
- a 2.98 GB 8 h 20 min workshop stream (`yt-conf-icml2024-mechinterp`, V4);
- 11-hour FLAC nights from one recorder (`birdvox-full-night`, `birdvox-full-season`, A4);
- a 7.9-hour WAV (`apsaa`, A2);
- a whole 24-hour day of a room as 1,441 one-minute files (`sins`, A1), which joins into one 11 GB WAV.

True file times beyond the name: the Bristol dashcam files carry a per-second UTC track (V2), and MEVA's names give
local start and end times (V1).

## Notes for fetching more

Speeds from this machine on 4 November 2026, with five searches downloading at once:

| Host | Speed per connection | What helped |
| --- | --- | --- |
| Google Cloud Storage (DROID, Rico, Android in the Wild, LILA) | about 15 MB/s, measured once before the searches | nothing needed |
| MEVA's S3 bucket | about 4 MB/s | lists without signing |
| Hugging Face | about 1 MB/s by range, faster whole | the `hf` CLI for whole repos; range requests for single zip members |
| Zenodo | 0.2-0.9 MB/s, up to 3.8 MB/s late at night | read a zip's central directory by range, then fetch only the members needed, checking each CRC32 (`D:\bench-data\tools\fetch\audio-long\zipget.py`, `sins_fetch.py`); mirrors (FSD50K from `Fhrozen/FSD50k`, checked against Zenodo file by file) |
| Internet Archive | 0.4-0.9 MB/s | several connections; IA's own MD5 list to check |
| AMI and ICSI (Edinburgh) | 0.06-0.5 MB/s | `diarizers-community/ami` on Hugging Face for AMI audio (not fetched) |
| data.bris.ac.uk (EPIC-KITCHENS) | about 0.25 MB/s, dropped connections | resume |
| Flickr originals | HTTP 429 after 128 files | stopped |

**YouTube:** the installed yt-dlp (2026.02.21) got only 360p or "not available". The current release worked with no
sign-in and no bot check in about 60 requests, offering every format up to 4K:

```bash
uvx --from "yt-dlp[default]" yt-dlp --js-runtimes node <url>
```

Every YouTube file is YouTube's re-encode. Internet Archive community-TV items keep the original files.

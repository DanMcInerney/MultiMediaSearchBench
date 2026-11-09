# The MMSB catalogue

Made by `mmsb catalogue` from the scenarios themselves, at scale `standard` and seed 1. Examples are from the `dev` split. Every answer is computed from the source's own labels and graded by a fixed rule.

## What people ask

Eight categories, by what the person wants, whatever the media. "It isn't there" is no category: about a quarter of the questions are about things that never happened, and read exactly like the others.

| Category | A person says | The answer |
| --- | --- | --- |
| Find it | "Find the photo of the parking-level sign." | one file or moment |
| Find every one | "Show me every clip with the cat." | a set of files, moments or dates |
| When | "What time did the van come? When did I last see it?" | a time or date |
| Did it happen | "Did the smoke alarm go off? Was there ever a bear?" | yes and when, or not there |
| How many, how long | "How many times did the dog bark? How long did it run?" | a number |
| What did it say or show | "What was the total? What deadline did they give?" | a short value, read or heard |
| Who or which | "Who suggested the price? Which camera saw it first?" | a person, file or place |
| Before and after | "What happened right after the alarm? Was the parcel before the rain?" | an order, or a neighbouring event |

## The media, by shelf

### Video

| ID | Shelf | Life area | The pile | Where it is from | Size (standard) | Questions (absent) |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | Security cameras | Work | three cameras of one site, over an hour of continuous footage | MEVA (scripted activity at a real site) | 6 files, 3.5 h | 22 (6) |
| V2 | Dashcam event clips | Home and family | staged: about a hundred short clips of crashes, near misses and ordinary drives, cut from two YouTube compilations of many people's dashcams and presented as one card, with invented dates and times | Australian Roads Dashcam Driving | 107 files, 0.4 h | 22 (5) |
| V3 | Football match clips | Fun and hobbies | the clips of three amateur matches in date folders, thirty seconds each, from a virtual follow camera | SoccerTrack v2 follow-cam clips (a virtual camera) | 525 files, 4.4 h | 23 (6) |
| V4 | Work meetings | Work | twelve recorded design meetings of three teams, a corner camera and the sound of the room | AMI Meeting Corpus (role-played) | 12 files, 6.6 h | 24 (5) |
| V8 | Smart-glasses days | Home and family | days of one person's wearable video, with speech | EgoLife | - | 50 (0) |
| V9 | Lecture recordings | School | one course's term of recorded lectures, two a week | MIT OpenCourseWare 14.01 | 26 files, 20.4 h | 23 (6) |
| V10 | Phone clips at home | Home and family | about 3,000 short phone videos of people at home, acted out from scripts, named as a phone names them | Charades (acted) | 3,000 files, 24.0 h | 23 (5) |

### Image

| ID | Shelf | Life area | The pile | Where it is from | Size (standard) | Questions (absent) |
| --- | --- | --- | --- | --- | --- | --- |
| I1 | Phone camera roll | Home and family | one person's phone photos over three years: food, posters, passes, sights, the cat | ATM-Bench (one person's phone) | 3,750 files | 24 (6) |
| I2 | Phone screenshots | Home and family | one Screenshots folder that mixes the screens of six kinds of phone (Pixel 3, 4, 4 XL, 5 and 6 and a custom device): thousands of screens of the browser and the apps in use | Android in the Wild | 3,071 files | 23 (5) |
| I3 | Trail camera | Fun and hobbies | one camera's memory card, nearly six months of triggers, most of them empty | Caltech Camera Traps | 4,256 files | 23 (6) |
| I4 | Building inspection photos | Work | an inspector's close-ups of walls in a folder for each job | BD3 building defects | 577 files | 24 (6) |
| I7 | Receipts and bills | Work | a folder of scanned receipts and bills that merges Malaysian shop receipts (ringgit) and Indonesian cafe receipts (rupiah), renamed scan_0001.jpg onwards | ICDAR 2019 SROIE, CORD v2 | 1,391 files | 22 (5) |
| I8 | Course notes and slides | School | a student's folder of saved pages of lecture notes and slides, page_0001.jpg onwards | MIT OpenCourseWare course PDFs | 439 files | 23 (5) |
| I9 | Nature photos | Fun and hobbies | four years of one person's walks: plants, birds, insects and animals, each named | iNaturalist (one observer) | 3,190 files | 24 (6) |

### Audio

| ID | Shelf | Life area | The pile | Where it is from | Size (standard) | Questions (absent) |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Room recorder | Home and family | a whole day from a living-room microphone, in one file | SINS database | 1 file, 24.0 h | 20 (5) |
| A3 | Research group meetings | Work | ten weekly meetings of one research group over thirteen months, sound only | ICSI Meeting Corpus | 10 files, 10.0 h | 24 (5) |
| A4 | Field recorder | Fun and hobbies | one night of a recorder in a field, hourly files, mostly quiet sky with faint calls | BirdVox-full-night | 11 files, 10.9 h | 18 (4) |
| A6 | Recorded lectures | School | a phone's recordings of one course, a class at a time through a term | MIT OpenCourseWare 6.0002 | 15 files, 12.2 h | 23 (6) |
| A7 | Podcast backlog | Fun and hobbies | the episodes of one interview podcast, kept in the order they came out | Chalk Radio (MIT OpenCourseWare) | 48 files, 18.2 h | 23 (6) |
| A8 | Music library | Fun and hobbies | thousands of short tracks of independent artists, in album folders, with their tags taken off | Free Music Archive (fma_small) | 2,461 files, 20.5 h | 20 (5) |
| M3 | Voicemails | Home and family | over a thousand robocalls from honeypots and voicemails of 31 FTC cases, in one folder, with invented times over nine weeks | FTC robocall recordings (machine transcripts) | 1,295 files, 8.9 h | 23 (5) |
| M4 | Sound-effects library | Fun and hobbies | thousands of unnamed clips in numbered folders | FSD50K | 6,123 files, 16.5 h | 21 (5) |

### Mixed

| ID | Shelf | Life area | The pile | Where it is from | Size (standard) | Questions (absent) |
| --- | --- | --- | --- | --- | --- | --- |
| V7 | Developer's week | Work | a synthetic week of screen recording, notes, logs, phone photos, porch camera and microphone | Made by the benchmark | 1,572 files | 31 (9) |

## Balance

### Shelves, by media and life area

|  | Home and family | Work | School | Fun and hobbies | All |
| --- | --- | --- | --- | --- | --- |
| Video | 3 | 2 | 1 | 1 | 7 |
| Image | 2 | 2 | 1 | 2 | 7 |
| Audio | 2 | 1 | 1 | 4 | 8 |
| Mixed | 0 | 1 | 0 | 0 | 1 |

### Questions, by category and media

| Category | Video | Image | Audio | Mixed | All | Absent |
| --- | --- | --- | --- | --- | --- | --- |
| Find it | 15 | 24 | 14 | 0 | 53 | 17 |
| Find every one | 15 | 40 | 32 | 0 | 87 | 22 |
| When | 30 | 23 | 32 | 12 | 97 | 28 |
| Did it happen | 21 | 22 | 30 | 0 | 73 | 33 |
| How many, how long | 31 | 25 | 31 | 5 | 92 | 15 |
| What did it say or show | 47 | 18 | 15 | 14 | 94 | 6 |
| Who or which | 19 | 5 | 11 | 0 | 35 | 1 |
| Before and after | 9 | 6 | 7 | 0 | 22 | 0 |

### Questions, by what must be perceived

| Cue | Questions |
| --- | --- |
| sight (seen) | 188 |
| sound (heard, not speech) | 86 |
| speech (heard, words) | 136 |
| text (read on screen or paper) | 90 |
| mixed (several of these) | 53 |

553 questions in all, 122 of them about things that are not there (22%).

V8 is filed by the question types of its dataset, not question by question, so some of the questions it counts under "What did it say or show" ask who or when.

## Questions, by category

### Find it

- *video, V10:* "Find the clip of me fixing a doorknob." Answer: VID_20250924_205530.mp4
- *video, V2:* "Which clip was filmed in a car park?" Answer: 2025_1218_113243_044.MP4
- *video, V4:* "In which meeting did anyone talk about a mango?" Answer: meeting_20050309_140521.mp4
- *image, I2:* "Find the screenshot that says "find key contacts at top companies"." Answer: Screenshot_20260925-120213.png
- *image, I7:* "Find the receipt from Kfa Supply on 27 May 2018." Answer: scan_0799.jpg
- *image, I1:* "Recall the Guangzhou Tower photo I took in 2023." Answer: not there
- *audio, A7:* "In which episode did they talk about Texas?" Answer: episode_31.mp3

### Find every one

- *video, V2:* "Show me every clip with a crash at night." Answer: 2026_0103_133134_086.MP4, 2026_0104_165042_093.MP4, 2026_0109_060807_103.MP4
- *video, V3:* "Find every clip with a goal in the match on 8 March." Answer: FC_185800.mp4, FC_201330.mp4
- *video, V9:* "Which lectures mention Elon Musk?" Answer: lecture_2023-09-20.mp4, lecture_2023-09-25.mp4
- *video, V2:* "Show me every clip with a crash in a car park in the rain." Answer: not there
- *image, I1:* "I remember the poster session venue for NeurIPS 2023 was huge. Recall photos of the venue." Answer: IMG_1579.jpg, IMG_1580.jpg, IMG_1581.jpg, IMG_1582.jpg
- *image, I2:* "Show me every screenshot from when I googled "Where can I buy a nice beach blanket"." Answer: Screenshot_20260616-211554.png, Screenshot_20260616-211601.png
- *image, I4:* "Give me every photo of algae on the Chapel Lane job." Answer: IMG_0189.jpg, IMG_0190.jpg
- *image, I2:* "Show me every screenshot from when I googled "What is the capital of the United Kingdom"." Answer: not there
- *audio, A3:* "Which meetings mention forced alignment?" Answer: rec_20000216_1600.wav, rec_20001207_1450.wav, rec_20010111_1440.wav, rec_20010329_1400.wav
- *audio, A6:* "Which lectures mention the normal distribution?" Answer: recording_20161109_140626.m4a, recording_20161114_140737.m4a, recording_20161116_140558.m4a, recording_20161212_140724.m4a
- *audio, A7:* "Which episodes mention mental health?" Answer: episode_23.mp3, episode_25.mp3, episode_29.mp3
- *audio, A8:* "Find every rock track in the folder Album 057." Answer: not there

### When

- *video, V1:* "When did someone last close a car door on cam2?" Answer: 2018-03-12 11:01:53
- *video, V1:* "When did two people last hug on cam1?" Answer: 2018-03-12 11:01:39
- *video, V1:* "When did someone first buy something on cam2?" Answer: not there
- *image, I1:* "I saw a hot air balloon that says "Bailey". When is that" Answer: 2022-07-07
- *image, I3:* "When did my trail camera first photograph a dog?" Answer: 2013-10-17 16:43:35
- *image, I7:* "On which day did I shop at Nsk Trade City-selayang?" Answer: 2017-10-27
- *image, I1:* "I saw an ad for pokemon society, when is that?" Answer: not there
- *audio, A1:* "At what time yesterday, between 18:30 and 22:00, did I stop watching TV?" Answer: 21:56:56
- *audio, A1:* "At what time yesterday, between 14:00 and 15:30, did a visitor arrive?" Answer: 14:59:20
- *audio, A1:* "At what time yesterday, between 14:30 and 17:30, did a visitor leave?" Answer: 15:08:42
- *audio, A1:* "At what time yesterday, between 04:00 and 08:00, did I stop watching TV?" Answer: not there
- *mixed, V7:* "At what time did the last deploy of the web service finish on Saturday 7 November, in the morning (06:00 to 12:00)?" Answer: 11:28:56
- *mixed, V7:* "At what time was a parcel left at the front door on Friday 6 November, in the afternoon (12:00 to 18:00)?" Answer: 12:02:53
- *mixed, V7:* "At what time was a parcel left at the front door on Friday 6 November?" Answer: 12:02:53
- *mixed, V7:* "At what time was a parcel left at the front door on Sunday 8 November?" Answer: not there

### Did it happen

- *video, V10:* "Did I ever film myself cleaning a mirror? If so, when?" Answer: 2025-08-20 12:57
- *video, V2:* "Did the dashcam ever catch a crash at a roundabout in the rain? If so, which clip?" Answer: not there
- *image, I2:* "Did I ever google "What's the weather like in Moscow"? If so, which was the first screenshot of it?" Answer: Screenshot_20260830-143934.png
- *image, I4:* "Did I photograph any peeling paint on the Park Row job? If so, which photo was the first?" Answer: IMG_0239.jpg
- *image, I8:* "Did my algorithms notes ever get to chaining? If so, on which page?" Answer: page_0213.jpg
- *image, I1:* "Did I ever take a photo of the sunrise from the plane to Cornwall? If I did, when?" Answer: not there
- *audio, A4:* "Did the recorder pick up any flight call between 04:09 and 04:11? If so, when was the first one?" Answer: 04:10
- *audio, A7:* "Did they ever mention climate change? If so, when was the first time?" Answer: 2020-02-19 10:21:22
- *audio, A8:* "Is there any rock in the folder Album 074? If so, which track is the first one?" Answer: track_1960.mp3
- *audio, A3:* "Did someone ever mention an earthquake? If so, when was the first time?" Answer: not there

### How many, how long

- *video, V1:* "How many times did someone close a car door on cam1 between 10:15 and 10:30?" Answer: 3
- *video, V10:* "How many clips are there of me getting dressed?" Answer: 18
- *video, V2:* "How many ordinary drives did the dashcam catch on a highway?" Answer: 2
- *video, V1:* "How many times did a car reverse on cam2 between 10:20 and 10:30?" Answer: not there
- *image, I3:* "On how many different days did a bobcat come past my trail camera?" Answer: 27
- *image, I9:* "On how many different days did I photograph a Common St. John's Wort?" Answer: 9
- *image, I3:* "On how many different days did a rabbit come past my trail camera?" Answer: 37
- *image, I2:* "How many screenshots did I take when I googled "What's the US dollar exchange rate against the Brazilian Real"?" Answer: not there
- *audio, A1:* "How many times did the vacuum cleaner run yesterday?" Answer: 2
- *audio, A3:* "In how many meetings did someone mention lapel mikes?" Answer: 7
- *audio, A4:* "About how many flight calls did the recorder pick up between 09:02 and 09:04?" Answer: 10
- *mixed, V7:* "How many times did the payments service log ERR-985 on Sunday 8 November, in the evening (18:00 to 24:00)?" Answer: 2
- *mixed, V7:* "How many times did the payments service log ERR-842 on Friday 6 November?" Answer: 6
- *mixed, V7:* "How many photos of a red car did I take on Friday 6 November?" Answer: 4
- *mixed, V7:* "How many times did the payments service log ERR-996 on Saturday 7 November?" Answer: not there

### What did it say or show

- *video, V2:* "What time of day was the crash at a roundabout?" Answer: Night
- *video, V4:* "In the fourth meeting on Wednesday 9 March 2005, what did the project manager say at the end that each remote would cost to operate, in euros?" Answer: 12.20
- *video, V8:* "Who usually pays when going out today?" Answer: Me
- *image, I1:* "I remember seeing a poster about HDR with a very strong visual effect. Can you help me find which poster it was?" Answer: single image hdr reconstruction
- *image, I2:* "What was the humidity in the weather I looked up for Chicago?" Answer: 58
- *image, I7:* "How much did I pay for the Peduli Disabilitas, in rupiah?" Answer: 200
- *audio, A3:* "In the meeting on Wednesday 2 February 2000, how much did they say they could spend without worrying about it?" Answer: a thousand dollars
- *audio, A6:* "In the first lecture the professor described the finger exercises of the problem sets. If one takes you longer than how many minutes should you let the staff know?" Answer: 15
- *audio, A6:* "In the lecture on predicting who survived the Titanic, how many passengers did the professor say died?" Answer: 812
- *mixed, V7:* "What is the code of my gym locker that I noted on Sunday 8 November, in the morning (06:00 to 12:00)?" Answer: 5431
- *mixed, V7:* "What is the combination of my bike lock that I noted on Friday 6 November?" Answer: 7113
- *mixed, V7:* "What was retention_days set to in app.toml at 08:44 on Saturday 7 November?" Answer: 28
- *mixed, V7:* "What is the Wi-Fi password that I noted on Sunday 8 November, in the evening (18:00 to 24:00)?" Answer: not there

### Who or which

- *video, V10:* "Which room was I in when I was washing some clothes?" Answer: Laundry room
- *video, V8:* "Who just helped Katrina put the flowers on the table" Answer: Me
- *video, V10:* "Which room was I in when I was fixing a doorknob?" Answer: Bathroom
- *audio, A7:* "Who was the guest on the episode about the cloud chamber?" Answer: Anne White
- *audio, A8:* "Which folder has the most electronic tracks?" Answer: Album 033
- *audio, M3:* "Who did the voicemail I got on Wednesday 28 October at 15:48 say it was from?" Answer: paypal

### Before and after

- *video, V10:* "In the clip of me cleaning a mirror, was I standing up before or after cleaning a mirror?" Answer: after
- *video, V3:* "In the clip from 19:57:30 on 8 March, was the tackle before or after the shot?" Answer: after
- *audio, A3:* "In the meeting of Thursday 2 November 2000, did someone bring up Eurospeech before or after a Markov model?" Answer: after

# MultiMediaMemoryBench: vision

Start here. This file says what MMMB is for and what it must stay true to. How it is built is in [DESIGN.md](DESIGN.md).

## The goal

**MMMB tells people how well an AI agent can find things in their own large piles of video, pictures and sound, and which
search tools make it better, judged on the media people really keep and the questions they really ask.**

People keep footage from security cameras, phones, dashcams, meetings, labs and field recorders, and they rarely look at
it again, because finding one moment in it takes hours. An agent that can be told "find when the parcel was taken" or
"find the voice memo where I said the gate code" changes that. MMMB measures how close agents and their tools are to doing
it well.

## What the goal asks for

1. **Real use, not puzzles.**
   - The media is what people really have: a security export, a camera roll, a night of audio, a folder of voice memos. It
     comes at the size and in the mess it really has.
   - The questions are ones people really ask, in their own words.
2. **Every kind of pile:**
   - big video files;
   - thousands of images;
   - long recordings;
   - hours of audio spread over many small files;
   - and mixes of these.

   Speech and sound count as much as pictures, even where today's tools are weak at them.
3. **Neutral.** Any agent and any search system can be measured, and none is favoured. MMMB belongs to no product. The
   agent working alone with the files is always the baseline.
4. **Right means shown to be right.** An answer counts only when it points to where in the media the answer is. Every
   answer is known in advance and checked by fixed rules, not by another model's opinion.
5. **"It isn't there" is an answer.** Questions about things that never happened matter as much as questions about things
   that did.
6. **Cost is part of the result.** Accuracy is reported together with:
   - the time to an answer;
   - the money spent;
   - the local computing used;
   - the wait before a pile can be searched at all.
7. **Honest numbers.**
   - Results say how sure they are, and small differences are called unproven.
   - Nothing is tuned on the questions it is scored on.
   - Shortcuts and leaks that would let a tool score without doing the task are hunted down and closed.
8. **Private media stays private.** People's own recordings can be used to test with. They never leave their machine and are
   never published.
9. **Anyone can run any part of it.** One question or all of them, on one ordinary computer, at a cost a person can afford,
   with results others can reproduce.
10. **Room to grow.** The questions stay hard for today's best agents and tools, so that real progress shows as a better
    score.

## Scope

**In:**
- finding things in personal and small-team media: home, phone, work, lab and field recordings;
- answering when, what, which, whether, how many, and "find every one".

**Out, unless decided otherwise:**
- making or editing media;
- describing or summarising media when nothing is being searched for;
- web-scale search over other people's media;
- answering live, while the recording is still being made;
- grading answers with another model.

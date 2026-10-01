# Reading a long document: four modes, measured

docuzent can read a long document - a book - for a question four ways (`docuzent-read`). This page says what each
does, why it was built the way it was, and what each costs and gets right on real books. Every number here comes
from `docuzent-eval` on one machine (below); where something was not measured, it says so.

Tracking issue: [#39](https://github.com/no-mans-code/docuzent/issues/39).

## The four modes

| mode | learning a book | a question |
|---|---|---|
| **rag** | cut into chunks; indexed by words (BM25) and meaning (embeddings) | the best 8 chunks, as they are, to the answer |
| **rag-expanded** | ... and every chunk described by the model: where it sits in the story, the people in it, its setting and objects, its facts, the questions it answers | the best chunks, found through any of those - answered from the chunks' own text |
| **rag-kv** | the expanded index, **and** every part's KV state saved | the expanded index's passages first, in the document's own words; then the parts they point at (at most 4), restored from their saved state and read whole, to enrich them |
| **kv** | every part's KV state saved | every part restored and scored (one token each); every relevant part read closely |

A *part* is up to 4,000 tokens of the document, cut between paragraphs - one context window's worth, whose KV state
llama.cpp saves to disk and restores in about a tenth of a second (instead of the ~2 seconds re-reading it takes).
A *chunk* is about 1,500 characters, never crossing a part, so every chunk points back to exactly one saved part.

### Why these designs

- **Hybrid search.** Words find exact names and rare terms ("Norbert", "Ridgeback", "vault seven hundred and
  thirteen"); vectors find meaning ("how did he feel" finds "wept"). Each ranks chunks by their best-matching
  entry; the rankings are fused by reciprocal rank (k = 60), which needs no tuning of score scales.
- **Expansions (Mode 2) follow published work**, each for a way a question can miss a chunk:
  - *Contextual Retrieval* (Anthropic, 2024) - each chunk prefixed with a sentence situating it in the document;
    reported 49% fewer failed retrievals, 67% with reranking. Here the model writes that sentence *with the chunk's
    whole part resident* (its saved KV state), so it sees the context at the cost of a restore.
  - *doc2query* (Nogueira et al., 2019) - the questions a passage answers, indexed.
  - *Dense X Retrieval* (Chen et al., 2023) - the atomic facts (propositions) it states.
  - the entities of *GraphRAG* (Edge et al., 2024) - people, places, objects, events - without the graph.
  The expansions only *find* a chunk; the answer is always written from the chunk's own words.
- **Mode 3: expanded RAG first, saved parts to enrich** (the project author's design). The first version handed the
  answer only the model's readings of the whole parts the index pointed at. Those readings are a model's words, not
  the book's; the expanded index's passages are the book's own, and precise. So the passages go first, verbatim, and
  the whole-part readings follow as enrichment. (The first version's one "miss" on the Gita turned out to be the
  grading: its answer - "neither born nor can it die" - was right, in words the check did not expect. The check was
  widened and every run re-graded from its saved answers - `eval/rescore.py`.)
- **Chunking** is measured two ways: plain (paragraphs up to a size) and guided (the model, with the part resident,
  marks where scenes and topics change).

### What "altered" (expanded) RAG is

Plain RAG can only find a chunk by what the chunk itself says. Questions are rarely asked in the book's words:

- *"How did Harry set the snake free?"* - the chunk where it happens says *"the glass front of the boa
  constrictor's tank had vanished"*; nobody "sets" anything "free".
- *"What does Krishna say happens to the soul at death?"* - Arnold's translation says *"Nay, but as when one layeth /
  His worn-out robes away ... So putteth by the spirit / Lightly its garb of flesh"*: neither "soul" nor "death".
- *"Who owns Fluffy?"* - the line that answers it is *"he's mine - bought him off a Greek chappie"*, said by a "he".

Altered RAG keeps the chunks exactly as they are and adds, for each one, **other ways to find it**, written by the
model while the chunk's whole part (~4,000 tokens) is in its view:

| entry | what the model writes | the question it catches |
|---|---|---|
| context | one sentence placing the chunk: who, where, when, what led to it - stored *with* the chunk's text | questions that use the situation, not the words ("at the zoo") |
| people | every named person in it, and what they do or say | "who", and pronoun-only chunks ("he's mine" -> Hagrid) |
| setting | place, time, objects, creatures, events | "where", "when", "what happened to the ..." |
| facts | the facts it states, short, names and numbers exact | factual and numeric questions in other words |
| questions | three questions it answers | the reader's phrasing itself (doc2query) |

Every entry is searched by words and by meaning like a chunk, and **every entry points back to its chunk**. Only the
chunk's own words ever reach the answer: an expansion that is wrong can make a chunk *found* when it should not be,
never put a wrong word in an answer. The cost is model time when a document is learned - roughly one call per three
chunks (measured below) - and a few times more index on disk, still a thousandth of the KV states.

Why the part must be in view: a chunk alone often cannot be described ("he went back up the stairs" - who? which
stairs?). With the part's saved KV state restored (~0.1 s), the model reads the chunk as part of its chapter. This
is the step that cannot be done cheaply without saved KV states - without them, each batch would re-read 4,000
tokens. (On Ollama, which saves nothing, it does re-read them; Ollama does keep the last prompt's prefix, so the
chunks of one part, described one after another, share most of that cost.)

### Off the leash

Every mode answers *on the leash* by default: from the passages alone, and "the document does not say" when they do
not. That is what makes the answers checkable - and what makes a question the book cannot answer a dead end.

**Off the leash** lifts that one rule for an answer (`answer::answer_leash(.., true)`; in thebook, the chain button
beside the microphone). The answer:

1. still starts from the passages the mode found - the document comes first;
2. may reason step by step and work numbers through (it always reasons before answering, when the model can);
3. may use what the model knows beyond the document where the document is not enough;
4. **must say which parts are which** - "the document says ...", "beyond the document ..." - and, for a story, say so
   before going further into it than the passages do (no unannounced spoilers).

What it is for: a numerical the book teaches the method for but does not work; a "why" the book does not explain;
connecting the book to the world. What it is not for: checking what a book says - on the leash is the honest mode
for that, and the evaluation grades both (a question can say what a right answer looks like off the leash: for one
the book cannot answer, "the document does not say" is right on the leash and wrong off it).

### Reasoning: auto, always, never

A model that can reason (Ollama reports a `thinking` capability; a llama.cpp model's chat template has `<think>`) can
work a question through before it answers - better on sums and comparisons, and up to 1,800 tokens slower. Whether it
does is a setting (`answer::Think`; `answer_with(.., think)`; `docuzent read --think`; in the evaluator,
`mode:think` / `mode:nothink`):

| setting | reasons through | for |
|---|---|---|
| **auto** (default) | counting, adding, comparing and ordering questions (`text::needs_thinking`), and every answer off the leash | most questions: a one-line answer is right, and the arithmetic that goes wrong without reasoning (a House Cup tie broken by ten points, then forgotten) gets it |
| **always** | every answer | hard questions the word list does not catch - a numerical worded as a story ("a stone is dropped from 80 m...") |
| **never** | nothing, even off the leash | speed; a model whose reasoning wanders or never ends |

On a model that cannot reason, every setting answers straight - asking it to would only ask twice. What always and
never gain and cost is measured below.

### Where things are kept

- **An index is stored per document, separately from the KV store, and is not size-bounded.** It is small: text and
  vectors, about a thousandth of the document's saved KV states (measured below). There is nothing to gain from
  evicting it, and rebuilding one with expansions costs model time.
- **Saved KV states** stay in the size-bounded, least-recently-used store; the bound is set at deploy time (any size -
  300 GB is fine if the disk has it). Evicting a part is always safe: it is simply read again when next needed.
- **A merged corpus** (thebook's crossovers - two books as one) has no index of its own yet: each document's index is
  searched on its own and the results interleaved by rank (`search_each`), so every document is represented. A
  single index across documents is future work.

### When the model changes

- **Saved KV states** belong to the exact weights and window that made them: a new model reads every part again.
- **Expansions** are the model's writing: they are made again by the new model (`Index::made_by` records who).
- **Embeddings** belong to the embedding model, not the chat model, and are made again only when it changes.

## How it is measured

`docuzent-eval` learns a book in each mode and answers a set of checked questions (`eval/sets/`) with the same
plain answerer, so the mode is what differs. Every answer is checked against patterns: facts that must appear
(located in the book's text first), and things a faithful answer must never say - outside knowledge, spoilers from
later books, filler. Each set has questions the book cannot answer; the right answer to those is "the document does
not say".

- **Harry Potter and the Philosopher's Stone** (~455,000 characters, 28 parts): a long novel the model partly knows.
  8 *tuning* questions (written after seeing an earlier version fail) and 9 *held-out* (written, and their answers
  located in the text, before any answer was seen).
- **The Song Celestial (Bhagavad-Gita)**, Edwin Arnold's translation (~124,000 characters, 10 parts): a short,
  old-worded teaching poem. 12 held-out questions.
- **The Subtle Art of Not Giving a F\*ck** (~318,000 characters): conversational non-fiction. 14 held-out questions.
- **NCERT Physics, Class 11** (~788,000 characters, both volumes): a textbook - facts, worked examples, numericals
  the book does not work, and questions beyond it. 20 held-out questions.

The books are not in this repository.

Machine: one RTX 5080 Laptop GPU (16 GB), Qwen3-14B Q4_K_M in llama.cpp (12,288-token window, 8-bit KV cache),
and smaller models through Ollama; embeddings by nomic-embed-text v1.5 (Q8) in llama.cpp **on the CPU**. Each run records the model's plain generation
speed before and after it; a run whose speed fell is marked, because something else on the machine was eating into
its timings.

## Results

Qwen3-14B (Q4_K_M) in llama.cpp, 12,288-token window. *Score* is questions right; *held out* the ones written,
and their answers located in the book, before any answer was seen. *Answer* is seconds per question, average (and
slowest). *Learning* is what the mode needs made once per book: saved parts, an index, expansions, embeddings.

**Which numbers to trust.** Harry Potter and the Gita were used while the engine was being built - their tuning
questions were written after seeing failures, and some rules came from fixing them. NCERT Physics and Subtle Art
were not: nothing was changed to suit them. Read the first two as "tuned on", the others as held out.

### Harry Potter and the Philosopher's Stone (tuned on)

455,142 characters, 28 parts. 17 questions (9 held out).

| mode | score | held out | answer | learning | on disk |
|---|---|---|---|---|---|
| rag | 13/17 | 8/9 | 9 s (46) | 1 min | 2.4 MB |
| rag, guided chunks | 12/17 | 8/9 | 7 s (54) | 5 min | 2.9 MB |
| rag-expanded | 14/17 | **9/9** | 8 s (45) | 64 min* | 9.5 MB |
| rag-kv | 13/17 | 8/9 | 15 s (29) | 64 min* | 9.5 MB + 9.3 GB |
| **kv** | **15/17** | 8/9 | 34 s (73) | **2 min** | 9.3 GB |

\* the first expansion and embedding (50 min) plus a repair of the passages it had left undescribed (14 min) - see
below.

What failed, and why (read from the answers, and from where the search ranked the passage that answers - with
`examples/probe.rs`):

- **Spread across a chapter** - "what obstacles guarded the Stone, in order?", "why was Harry made Seeker?" (the
  Remembrall catch and McGonagall's "I've found you a Seeker" are told in a scene that never says "position" or
  "chosen"; the plain index ranked it 114th, the expanded one 30th). Every top-k mode misses them; only
  **kv**, which reads every part, gets both.
- **Arithmetic** - the House Cup: Gryffindor's 472 tie broken by Neville's ten points. Every mode, kv included,
  had the deciding line in its evidence and stopped at the tie. A model limit, not a reading one.
- **A model's reading in place of the book's words** - rag-kv twice turned a right plain-RAG answer wrong: it gave
  a reason Hagrid was expelled that the book never gives (the readings of whole parts brought in "keeping a
  dragon"), and it said the book does not say why Harry was made Seeker, where rag-expanded - searching the
  same index - answered it.
- **Two things in one passage** - rag-expanded once answered Harry's wand with Voldemort's (the same scene names
  both).

### The Song Celestial - Bhagavad-Gita (tuned on)

124,089 characters (Edwin Arnold's verse translation), 10 parts. 12 questions, all held out.

| mode | score | answer | learning | on disk |
|---|---|---|---|---|
| rag | 11/12 | 3 s (6) | 21 s | 0.6 MB |
| rag, guided chunks | 11/12 | 4 s (12) | 3 min | 0.7 MB |
| **rag-expanded** | **12/12** | **3 s** (8) | 23 min* | 1.7 MB |
| rag-kv | 12/12 | 12 s (40) | 23 min* | 1.7 MB + 3.0 GB |
| kv | 12/12 | 25 s (85) | **26 s** | 3.0 GB |

\* 15 min of expansion plus an 8 min repair. On this old, verse-worded text, the meaning search was the weak half
of plain RAG (words alone found the evidence for 11 of 12 questions, words and meaning fused 9 - below); the
expansions' plain-English descriptions are what closed the gap.

### The Subtle Art of Not Giving a F\*ck (held out)

317,702 characters, 19 parts. 14 questions, all held out.

| mode | score | answer | learning |
|---|---|---|---|
| **rag** | **14/14** | **3 s** (6) | 1 min |
| rag, guided chunks | 14/14 | 3 s (9) | 3 min |
| rag-expanded | 14/14 | 7 s (28) | 32 min |
| rag-kv | 14/14 | 10 s (18) | 32 min |
| kv | 14/14 | 15 s (29) | 1 min |

Plain non-fiction that says what it means in the words a reader asks with: every mode is right, so the cheapest
is the one to use. (It also means this set is too easy to separate the modes - a finding about the set as much as
the book.)

### A bug the measurements found: expansions silently missing

The first expanded indexes had holes: 52 of Harry Potter's 464 chunks and 41 of the Gita's 105 had no
descriptions at all, three at a time - whole batches whose reply ran past its token limit, so the JSON was cut off
and the batch dropped without a word. Retrieval measured on the holed index made expansion look useless on Harry
Potter (12/17, no better than plain RAG); after the fix (finished descriptions kept from a cut-off reply, missing
passages described again on their own, saved indexes repaired rather than rebuilt) the same mode scored 14/17 with
every held-out question right. A newly built index (Subtle Art: 237 chunks) had none missing; Harry Potter's
repair left one passage the model would not describe even alone.

### Retrieval alone (no model): what reaches the answer

`docuzent-eval`'s `retrieval` tool asks, for each question, whether the text a search hands over contains what a
right answer must say - separating a retrieval miss from an answering one, in seconds. Questions whose evidence
was found complete, out of those the book can answer (expanded indexes after the repair):

| index | words + meaning (now) | words only | meaning only | + neighbouring chunks |
|---|---|---|---|---|
| Harry Potter, plain | 11/14 | 12/14 | 11/14 | 11/14 |
| Harry Potter, expanded | 12/14 | 11/14 | 12/14 | 12/14 |
| Gita, plain | 9/10 | 10/10 | 9/10 | **10/10** |
| Gita, expanded | 10/10 | 10/10 | 9/10 | 10/10 |
| Subtle Art, plain | 13/13 | 12/13 | 12/13 | 13/13 |
| Subtle Art, expanded | 13/13 | 13/13 | 13/13 | 13/13 |
| NCERT Physics, plain | 15/18 | 16/18 | 15/18 | **16/18** |
| NCERT Physics, expanded | 15/18 | 16/18 | 15/18 | **16/18** |

*Neighbouring chunks*: each found chunk handed over with the last 600 characters of the chunk before it and the
first 600 of the one after (same part). A chunk is cut at a size, not where a thought ends. It helped on three of
eight indexes - two of them the held-out textbook - and never hurt: a small, safe gain, tested end to end below.
Words alone beat the fusion on four indexes and lost on two: not adopted.

(A first run of this tool also counted the questions the book cannot answer - whose "evidence" is a phrase like
"does not say", which more text can contain by chance. It overstated the neighbours' gain on Harry Potter; the
tool now skips them.)

### NCERT Physics, Class 11 (held out)

788,081 characters - the two-volume textbook, 78 parts (23 GB of saved parts), 606 chunks. 20 questions: 6 facts,
8 worked examples the book solves, 4 numericals it does not (the question's own figures, the book's formulas), and 2
beyond the book (quantum entanglement, the Schrödinger equation - "the book does not say" on the leash, a real
answer off it).

| mode | score | answer | learning |
|---|---|---|---|
| rag | 18/20 | 4 s | 2 min |
| rag-expanded | 18/20 | 6 s | 74 min |
| rag-kv | 17/20 | 14 s | 74 min |
| **kv** | **19/20** | 38 s | **4 min** |
| rag-kv, always reasons | 18/20 | 22 s | 74 min |
| kv, always reasons | 17/20 | 46 s | 4 min |
| **rag-kv, off the leash** | **20/20** | 27 s | 74 min |
| kv, off the leash | 19/20 | 49 s | 4 min |
| **kv, off the leash, never reasons** | **20/20** | 38 s | 4 min |

- **On the leash, one numerical failed in every mode and passed in every one off it**: a 2 kg block pushed by 10 N
  for 5 s. The book gives F = ma and v = at; on the leash the answer read "use only these passages" as forbidding
  the question's own figures ("the document does not say what the speed of the block is"). The answer prompt now
  says a question's own figures may be worked through with what the passages say (measured below).
- **Reasoning on every answer did not help.** It moved which questions were right (two fixed, two broken) and
  cost 20-60% more time. *Auto* stays the default; *always* is a trade, not an upgrade.
- **Off the leash, reasoning was not needed either**: kv off the leash scored 20/20 answering straight and 19/20
  reasoning first (and was faster).

**A grading lesson.** The first grading scored kv 15/20 and suggested that its readings blur figures. They did not:
the model writes maths in LaTeX (`t = 4 \, \text{s}`), and a check for "4 s" could not read it; another check
missed "1 metre" written out. Every answer is now reduced to plain text before it is checked - the same for every
run - and the saved answers were graded again (`eval/rescore.py`). Nineteen answers on NCERT changed from wrong to
right (nine of them the seconds pendulum's "1 metre", in every run), none on the other books; the ones read back
were all right.

### Improvements, tested against the engine before them

Each was run on the same indexes and saved parts as the engine before it, so only the reading differed:

| change | where it was measured | result |
|---|---|---|
| a found chunk handed over with the edges of its neighbours (600 characters each side) | Harry Potter, rag | 13 -> 14/17: "why was Harry made Seeker?" now answered |
| Mode 3's whole-part readings labelled as notes, below the passages, the passage right where they differ | Harry Potter, rag-kv | 13 -> 14/17, held out 9/9: the invented reason for Hagrid's expulsion gone ("the passages do not specify the exact reason") |
| both | the Gita | rag 11 -> 11; rag-kv 12 -> 11 - two answers that say the same thing, one of which passed on a stray "does not" (a grading artifact, not a change) |
| a question's own figures may be worked through with the passages' formula, on the leash | - | not run end to end before testing stopped; the case it answers is the block numerical every on-leash mode failed and every off-leash one passed |
| a scan keeps a stable set of saved parts when a document is larger than the store | unit-tested (a 5-part scan over a 3-part store: 0 restores with LRU, 6 with the change) | not run on a book |
| expansions: what a cut-off reply finished kept, left-out passages described alone, saved indexes repaired | Harry Potter, Gita | holes of 11% and 39% closed; rag-expanded on Harry Potter 12 -> 14/17 |

### An issue in our own logic: passages cut off unread, and one window for every model

Testing small models showed two faults in docuzent itself, both fixed:

1. **Evidence beyond a fixed size was dropped without a word.** The answer prompt took passages up to 26,000
   characters and silently left out the rest (and thebook's step answers did the same at 24,000). A passage past
   the cut was never read - and the answer could then say "the document does not say" about what it says. Data a
   model never sees is the one thing a reader must not allow. Now, passages more than one call holds are **swept**:
   every one is read, in groups that fit, the sentences that bear on the question are copied out word for word,
   and the answer is written from those (sweeping again if they still do not fit; a passage longer than a call is
   split, never cut). `answer::fit_passages`; thebook's steps use it too.
2. **Every model was given the same sizes.** Parts of 4,000 tokens, 8 chunks per search, 26,000 characters per
   answer - sizes for a 12,288-token window, applied to every model, and a model with a smaller window refused
   outright. Now every size comes from the model's window (`Budget`): a 1,024-token window reads a document in
   parts of ~330 tokens and chunks of ~420 characters, and a 32,768-token window hands over 21 chunks per search
   instead of 8. At 12,288 tokens every size is exactly what it was, so the results above stand as measured.
   docuzent plans for at most **32,768 tokens** whatever a model offers - a hard limit, which
   `DOCUZENT_MAX_CONTEXT` can lower but not raise; the least window it reads with is 1,024.

The Qwen3-14B results above were measured before the fix and were not re-run. Where a question's evidence passed
26,000 characters (likeliest in kv mode, when many parts bear on it), passages past the cut were not read: those
scores are, if anything, lower than the engine now gives.

### Smaller models

The same questions, through Ollama, on the same RTX 5080 (with the 14B server stopped). Ollama saves no KV states,
so kv and rag-kv read every part they need afresh - slower than llama.cpp, the same answers. Generation speed is
the evaluator's own check (64 tokens).

**Is the window what we ask for?** Yes. docuzent asks Ollama for a window (`num_ctx`) of the model's own size,
capped at what it plans for; `ollama ps` while each model ran showed exactly that - 12,288 tokens, 100% on the GPU,
for qwen2.5:3b (2.6 GB), llama3.2:1b (1.9 GB) and qwen3:0.6b (1.9 GB) - and qwen3:0.6b at its forced 1,024-token
window read in parts of 333 tokens, as planned. yi:6b (a 4,096-token model) was refused under the old 8,192-token
floor; with every step now sized to the window, it would be read in parts of ~1,330 tokens instead.

| model | size | tokens/s | Gita (12): rag / rag-expanded / rag-kv / kv / kv off-leash | NCERT (20): rag / kv |
|---|---|---|---|---|
| *Qwen3-14B (llama.cpp)* | *9 GB* | *~55* | *11 / 12 / 12 / 12 / -* | *18 / 19* |
| **qwen2.5:3b** | 2.6 GB | ~230 | 9 / **10** / 8 / 6 / 9 | **17** / 13 |
| llama3.2:1b | 1.9 GB | ~500 | 5 / 5 / 5 / 3 / 6 | 7 / 8 |
| qwen3:0.6b, 1,024-token window | (not captured) | ~550 | **7** / 6 / 6 / 6 / 6 | **13** / - |
| qwen3:0.6b, 12,288-token window | 1.9 GB | ~550 | 6 / 6 / 6 / 4 / 4 | 12 / - (rag-expanded 12, rag-kv 10) |

(qwen2.5:3b's runs and llama3.2:1b's Gita runs were made before the window and truncation fixes below; at the
12,288-token window those change nothing for retrieval modes, but kv could then lose evidence past 26,000
characters. The testing was stopped before the planned re-runs - qwen2.5:3b on the Gita, and qwen3:0.6b's kv on
NCERT and its 32,768-token window.)

What this says:

- **The cheapest model worth using is qwen2.5:3b, with rag or rag-expanded.** On the held-out textbook it is one
  question behind the 14B (17 against 18 of 20) at twice the speed and under a third of the memory; on the Gita two
  behind (10 against 12). For looking things up in a clearly written document, a 3B model gives up little.
- **Below 3B it falls away.** llama3.2:1b answered 3-6 of 12 and 7-8 of 20 - often with the evidence in front of
  it ("the book does not mention Pascal's law", beside the passage stating it). qwen3:0.6b does better than the 1B
  (7/12, 13/20) but is not a faithful reader.
- **kv is a large-model mode.** It makes the model judge and summarise every part; the 14B does that best of all,
  small models worst of all (qwen2.5:3b: kv 6/12 against rag-expanded 10/12; 13/20 against 17/20 on NCERT, and 20x
  slower through Ollama).
- **rag-kv's readings hurt small models too** (qwen2.5:3b 8/12, qwen3:0.6b 10/20 against 12 for rag).
- **A small model is no worse - sometimes better - in a small window.** qwen3:0.6b scored as well or better at
  1,024 tokens as at 12,288 in every mode (kv 6/12 against 4/12): its reading suits short parts, and the window's
  memory falls with it. A bigger window does not make a small model a better reader.
- **Expansions need a model that can write them in the room it has.** At first qwen3:0.6b left 346 of the Gita's
  421 chunks undescribed at 1,024 tokens - not its fault: it was asked for a five-field description and cut off
  after ~250 tokens. Asked for what fits (a sentence and two questions), it described 419 of 421.

## Which mode, when

What the measurements say - with the caveat that each book has 12-20 questions, so a difference of one question
is noise and the guide follows only what held across books:

| your situation | use | why |
|---|---|---|
| plain, clearly written non-fiction | **rag** | every mode was right (Subtle Art 14/14); rag is 3 s an answer and learns in a minute |
| a story, or a question whose answer is spread across chapters ("in order", "why did...") | **kv** (with a model of ~14B) | the only mode that reads every part: best on Harry Potter (15/17), and the only one to list the Stone's obstacles in order |
| a textbook, manual, or anything with exact figures | **kv** or **rag-expanded** | NCERT: kv 19/20, rag and rag-expanded 18/20 on the leash |
| old, unusual or technical wording (verse, archaic English, jargon) | **rag-expanded** | the model's plain-English descriptions find what the words do not (Gita: 12/12 against 11/12) |
| a small model (3B and below) | **rag** or **rag-expanded** - never kv | kv asks the model to judge and summarise every part; small models do that badly (qwen2.5:3b: kv 6/12, rag-expanded 10/12) |
| speed matters most | **rag** | 1-4 s an answer at any size; no model call to search |
| you want the book's facts, checkably | on the leash (the default) | the answer says "the document does not say" rather than fill in |
| a numerical the book teaches but does not work, a "why" it does not explain | **off the leash** | NCERT: 20/20 off the leash against 17-19 on it |

**rag-kv** (expanded passages first, the parts they point at read whole after) was never the best mode on any
book: its whole-part readings are a model's words beside the book's, and they twice brought in something the book
does not say. Labelling them as notes (below the passages, "where they differ, the passage is right") fixed one of
the two on Harry Potter (the invented reason for Hagrid's expulsion); it stays a middle ground - more context than
RAG, cheaper than kv - rather than a default.

**Reasoning (`think`)**: leave it on *auto*. Forcing it on every answer moved scores both ways (NCERT: rag-kv
17 -> 18, kv 19 -> 17) and cost 20-60% more time; off the leash, answering straight scored as well as reasoning
first.

## How far to trust these numbers

- **Small sets.** 12-20 questions a book: one question is 5-8 points, so a difference of one is noise. The guide
  above leans only on gaps that held across books or were large (kv against RAG on small models; 1B against 3B;
  off the leash on the textbook).
- **Grading by patterns.** Each answer is checked for what it must and must not say. Three checks were found too
  strict or too loose while reading answers (a LaTeX-typeset "4 s", "1 metre" written out, a Gita answer that
  passed on a stray "does not"); the first two were fixed and every saved answer re-graded. A second, independent
  grader (a model judging, spot-checked by hand) is the next step.
- **The engine changed during the study** - the expansion repair, the window budget, the sweep. Each result says
  which engine it was measured on; the 14B was not re-run after the last fixes.
- **One machine, one run each,** and no off-the-shelf tool measured on the same questions yet.
- **The question writer is the engine's author.** Harry Potter and the Gita were also used to tune it; NCERT,
  Subtle Art and every small-model result are held out.

## What was learned building it

- **Keep embeddings off the GPU the model is on.** With nomic-embed-text loaded in Ollama next to the 14B model, the
  model's memory spilled past the card and Windows ran it in shared system memory: generation fell from ~45 to ~1
  token per second. On the CPU the embedder costs the model nothing (verified: GPU memory unchanged).
- **Measure the machine, not only the method.** A stray process spinning a CPU core made one run's per-question
  times ~10x worse; the speed check around each run is there to catch that.
- **A scan longer than the store gets nothing from LRU.** Mode 4 reads every part in the same order; on a document
  with more parts than the store holds, least-recently-used eviction removes each part just before the next scan
  wants it. A scan now keeps a stable set and turns the newcomer away (`KvStore::register_keeping`). Note also that
  a document gets at most half the store while others are in it - NCERT's 78 parts (23 GB) needed the others
  cleared or a larger budget.
- **Silent failures are the expensive ones.** Expansions dropped without a word, passages cut off without a word,
  a looping reply ending a whole run, a grading check reading LaTeX as wrong: each was found only because a number
  did not make sense, and each now either cannot happen or says so.
- **Size the request to the room.** A model asked for more than its window lets it write does not fail cleanly -
  it is cut off mid-object and nothing is kept. Every request is now sized from the window (`Budget`).

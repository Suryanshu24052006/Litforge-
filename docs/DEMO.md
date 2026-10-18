# Longer-trained demo

This is a separate, single-seed attempt to improve the generated text. The original three-seed comparison and its test scores are unchanged. Both Transformer runs use the same architecture and initialization seed. The larger training budget also extends the learning-rate schedule. Checkpoint selection uses validation BPB only.

| Model | Selected step | Training bytes at that step | Validation BPB ↓ | Generated words seen in training |
|---|---:|---:|---:|---:|
| original | 1,200 | 2,457,600 | 2.7639 | 74.9% |
| extended | 16,000 | 32,768,000 | 1.9879 | 95.7% |

The last column counts complete ASCII words of two or more letters that also occur in the training books. It excludes a possibly cut-off final word. This is a spelling proxy: real words can still form incoherent sentences. Raw word counts, replacement-character counts, and checkpoint hashes are in the [comparison record](../artifacts/demo/comparison.json). No test book is read here. Validation has already guided development, so this is not a new independent test result.

All fixed prompts and sampling seeds are declared in the [plan](../experiments/extended-demo.json). Both models use the same generation settings. Samples below are unedited; none were selected for fluency.

## Prompt: It was a quiet evening, and · sampling seed 42

### original

```text
It was a quiet evening, and being all that be
fore mor to you was con the of him not the simite to arrraie of the was
the be to of his the for them of and on of he cand for for she live her could
and in prie the she pertion suprt the was his dents. 

Mrs. Pirrf. Nonll
```

### extended

```text
It was a quiet evening, and proper, that she had been any
one worther of internations the most of hardly, and the expect shame
in their situation many of a thousand was readful some particular
interesting.   I was more than she was easy. It will soon be in in from
her
```

## Prompt: It was a quiet evening, and · sampling seed 123

### original

```text
It was a quiet evening, and whought with a lin
was of it with you dis acking ito sall tal to have was as he of cof
her mayming hawered do her in the could this his and the the his him soment to he
reardied of to of her she she withe explance to frobectle with he prose
```

### extended

```text
It was a quiet evening, and whose were now to be in his
way all the same of a voice.  The companion of a reglation of highly
mother appearance of her informance in a morning with she is a distress
regret of Mr. Collins was surprised to her as they because it may be
so
```

## Prompt: She looked at him and said, · sampling seed 42

### original

```text
She looked at him and said, but and to all and
her more pose wer acon the of him not the simite to arrraie of the was
the be to of his the for them of and on of he cand for for she live her could
and in prie the she pertion suprt the was his dents. 

Mrs. Pirrf. Nonll
```

### extended

```text
She looked at him and said, and all the asserting that he was excecessived
than he was the mind already, and her felt the days of their sisters of
the same too suppose and after her refusion, and a clination of sight
in the suppose of Miss Fairfax's business, for part
```

## Prompt: She looked at him and said, · sampling seed 123

### original

```text
She looked at him and said, whought with a lin
was of it with you dis acking ito sall tal to have was as he of cof
her mayming hawered do her in the could this his and the the his him soment to he
reardied of to of her she she withe explance to frobectle with he prose
```

### extended

```text
She looked at him and said, whose were no pleasure of it with his
daughters and her manner was a new man.

The contrarimation of Edward's life, the self pleasure of his mother,
she could always herself in the sister, which she was so many to prevent
that with her sist
```

## Prompt: The next morning, · sampling seed 42

### original

```text
The next morning, befir all that and
her mor to you was compened thing ther the imite to arrraie of the was
the be to of his the for them of and on of he cand for for she live her could
and in prie the she pertion suprt the was his dents. 

Mrs. Pirrf. Nonll
```

### extended

```text
The next morning, and all the asserting that he was expected to him.

In the most incommend this own complaint solitely and her effection,
and he could dispose to for Mrs. Ferrars.

Elton was all to a signs in much thrown the interest of an intimater
despect
```

## Prompt: The next morning, · sampling seed 123

### original

```text
The next morning, whould gring a the
was out sure all sto he cauld ito say and to be of he mastion.

She he her mighte were prom you when seep presion indicioulde affurcelly somed cartied
the impenced fore comp the it in she him to to frobectle with he prose
```

### extended

```text
The next morning, whose were no pleasure of it with you do not that
to say and to be a good want of giving the party of delight of the character
of their looking them, well in his continue of its little of company
the intention of speaking of his subject, an
```

## Prompt: I have always thought that · sampling seed 42

### original

```text
I have always thought that pof her to all and
her more pornowed acon the of him not the simeten of withir this
a was the be to of his the for them of and on of him was for
for she live her could and in prie the she pertion suprted the in
the encerat benice arrd; pand
```

### extended

```text
I have always thought that I have been as they were in poor with conversation,
and the house which will be the being the days in their situation many
of a thousand was readful some particular interesting.   I was more to
the profess.--My dear Miss Bates, with a reall
```

## Prompt: I have always thought that · sampling seed 123

### original

```text
I have always thought that whough gring at to
was out sure all she had to caivourate tall was of he mastion.

She had Mr. She Mars. But the tisted seep presion indister, and not of with and to he
reardied of to of her she she withe exelly the her be the with
fore to 
```

### extended

```text
I have always thought that when I had not learned to it; but you do
not think of a man who the greatest in of its handsome time without me;
but I knew had soon indifferent a must have been calling to sight, and
one complaisely in speaking of the time, or it many of y
```

## Reproduce

```bash
uv run --frozen python -m litforge prepare-books
uv run --frozen python -m litforge study --plan experiments/extended-demo.json --output runs/my-extended-demo --phase train
uv run --frozen python -m litforge export-demo --run runs/my-extended-demo
```

The full settings, training history, and output comparisons are saved in `artifacts/demo/`. The included inference model does not contain optimizer or RNG states; those stay in the local training directory.

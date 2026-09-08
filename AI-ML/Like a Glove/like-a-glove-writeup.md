# HTB Like a Glove — AI Challenge Write-up

## Challenge Description

> Words carry semantic information. Similar to how people can infer meaning based on a word's context, AI can derive representations for words based on their context too! However, the kinds of meaning that a model uses may not match ours. We've found a pair of AIs speaking in metaphors that we can't make any sense of!
>
> The embedding model is `glove-twitter-25`.
>
> The flag should be fully ASCII and starts with `htb{`.

The challenge provides a text file containing 84 analogy questions in this format:

```text
Like A is to B, C is to?
```

Example:

```text
Like non-mainstream is to efl, battery-powered is to?
```

At first, the relationships look meaningless. This is intentional.

The challenge is not asking us to interpret the words like a human. Instead, we need to reproduce the relationships using the same word-embedding model used by the challenge: `glove-twitter-25`.

---

## Key AI Concept: Word Embeddings

A word embedding converts a word into a numerical vector.

For example, conceptually:

```text
dog  -> [0.42, -0.15, 0.77, ...]
cat  -> [0.39, -0.10, 0.73, ...]
king -> [0.81,  0.35, 0.12, ...]
```

`glove-twitter-25` represents each token using 25 numbers.

Words that appear in similar contexts tend to have vectors that are mathematically close to each other.

A famous word-embedding analogy is:

```text
king - man + woman ≈ queen
```

This means vector arithmetic can sometimes capture relationships between words.

---

## Understanding the Challenge

Each challenge line has the form:

```text
A : B :: C : ?
```

To solve the missing value, we apply the same relationship from `A -> B` to `C`.

The vector formula is:

```text
answer_vector = vector(C) + vector(B) - vector(A)
```

Then we search the GloVe vocabulary for the token whose vector is closest to `answer_vector`.

In Python:

```python
target = model[c] + (model[b] - model[a])
answer = model.similar_by_vector(target, topn=1)[0]
```

The first returned token is the answer for that analogy.

---

## Why the Words Look Strange

The model used is:

```text
glove-twitter-25
```

It was trained on Twitter data, so its vocabulary includes:

- normal English words
- slang
- misspellings
- abbreviations
- foreign-language words
- Unicode text
- laughter such as `hahahaha`
- unusual tokens
- full-width characters

That explains strange challenge inputs such as:

```text
ajaajjajaja
nhahahahahaha
年度第
وشعب
３０
ｒ
```

The goal is not to understand those tokens manually.

The model already knows their positions in embedding space.

---

## Solver

### Requirements

Install the required Python packages:

```bash
pip install gensim numpy scipy
```

The first execution will download the `glove-twitter-25` model.

---

### `solve.py`

```python
import re
import unicodedata
import gensim.downloader as api

print("[*] Loading glove-twitter-25...")
model = api.load("glove-twitter-25")

flag_parts = []

with open("chal.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()

        match = re.match(
            r"Like (.+?) is to (.+?), (.+?) is to\?",
            line
        )

        if not match:
            print(f"[-] Could not parse line {i}: {line}")
            continue

        a, b, c = match.groups()

        try:
            # Solve:
            #
            # A : B :: C : ?
            #
            # ? = C + (B - A)
            target = model[c] + (model[b] - model[a])

            result, similarity = model.similar_by_vector(
                target,
                topn=1
            )[0]

            flag_parts.append(result)

            print(
                f"[{i:02}] "
                f"{a} : {b} :: {c} : {result} "
                f"({similarity:.6f})"
            )

        except KeyError as e:
            print(f"[-] Missing GloVe token on line {i}: {e}")

raw = "".join(flag_parts)

print("\n[+] RAW:")
print(raw)

# Convert compatibility/full-width Unicode characters to ASCII
flag = unicodedata.normalize("NFKC", raw)

print("\n[+] NORMALIZED:")
print(flag)

if flag.startswith("htb{"):
    print("\n[+] FLAG FOUND!")
```

---

## Running the Solver

Place these files in the same directory:

```text
chal.txt
solve.py
```

Run:

```bash
python3 solve.py
```

The script processes all 84 analogies.

---

## Example Recovered Lines

The beginning of the solver output was:

```text
[01] non-mainstream : efl :: battery-powered : htb (0.899523)
[02] sycophancy : بالشهادة :: cont : { (0.712081)
[03] беспощадно : indépendance :: rs : h (0.734182)
[04] ajaajjajaja : hahahahahahahahaahah :: ２ : ４ (0.932039)
[05] bahno : arbus :: duit : rm (0.770682)
[06] 잡히지 : ਮੈਂ :: 年度第 : ０n (0.840965)
```

Joining the recovered tokens produces the raw output.

---

## Raw Recovered String

```text
htb{h４rm０n１ou５_hymn_０f_h１ghd１m３ns１０n４l_subl１me_５ymph０ny_０f_num３r１cal_nuanc３_１n_tr３mend０u５_t４p３stry_０f_t３xtu４l_７r４n５f０rma７ion}
```

This looks almost correct, but it contains non-ASCII full-width Unicode digits.

Examples:

```text
４ -> 4
０ -> 0
１ -> 1
５ -> 5
```

The challenge description specifically tells us the final flag should be fully ASCII.

---

## Unicode Normalization

Python's `unicodedata.normalize()` can convert compatibility characters into their standard ASCII equivalents.

We use:

```python
unicodedata.normalize("NFKC", raw)
```

For example:

```text
２ -> 2
３ -> 3
４ -> 4
ｒ -> r
```

After NFKC normalization, the raw result becomes a valid ASCII HTB flag.

---

## Final Flag

```text
htb{h4rm0n1ou5_hymn_0f_h1ghd1m3ns10n4l_subl1me_5ymph0ny_0f_num3r1cal_nuanc3_1n_tr3mend0u5_t4p3stry_0f_t3xtu4l_7r4n5f0rma7ion}
```

---

## Attack / Solve Flow

```text
chal.txt
   |
   v
Parse each analogy
   |
   v
A : B :: C : ?
   |
   v
Calculate:
C + B - A
   |
   v
Find nearest token
in glove-twitter-25
   |
   v
Append recovered token
   |
   v
Repeat 84 times
   |
   v
Concatenate all tokens
   |
   v
Raw Unicode flag
   |
   v
NFKC normalization
   |
   v
ASCII HTB flag
```

---

## What I Learned

This challenge demonstrates several important AI concepts:

### 1. Embeddings

AI models can represent words as numerical vectors.

### 2. Semantic relationships

Relationships between words can sometimes be represented by directions in vector space.

### 3. Vector arithmetic

An analogy can be approximated using:

```text
C + B - A
```

### 4. Cosine similarity / nearest-neighbor search

After calculating a new vector, we search for the vocabulary token closest to that vector.

### 5. Model choice matters

The challenge specifically requires:

```text
glove-twitter-25
```

Using another embedding model may produce different answers.

### 6. AI representation is not human interpretation

The strange word relationships do not need to make sense to us.

They only need to make sense mathematically inside the embedding space.

### 7. Unicode matters

Visually similar characters may have different Unicode code points.

The challenge's ASCII hint tells us normalization is required.

---

## Short Summary

The challenge gives 84 word analogies.

For every line:

```text
Like A is to B, C is to?
```

we calculate:

```text
vector(C) + vector(B) - vector(A)
```

and find the closest word using `glove-twitter-25`.

The 84 recovered tokens are concatenated.

The result contains full-width Unicode characters, so we normalize it using:

```python
unicodedata.normalize("NFKC", raw)
```

This reveals the final flag:

```text
htb{h4rm0n1ou5_hymn_0f_h1ghd1m3ns10n4l_subl1me_5ymph0ny_0f_num3r1cal_nuanc3_1n_tr3mend0u5_t4p3stry_0f_t3xtu4l_7r4n5f0rma7ion}
```

---

## Challenge Category

```text
AI / Machine Learning
Word Embeddings
Vector Similarity
GloVe
Unicode Normalization
```

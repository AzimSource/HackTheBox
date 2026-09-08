HTB Like a Glove - AI Challenge Solver
=======================================

Overview
--------

This repository contains my solution for the Hack The Box AI challenge
"Like a Glove".

The challenge provides 84 strange-looking word analogies and tells us that
the embedding model is:

    glove-twitter-25

Each line looks like:

    Like A is to B, C is to?

The words usually do not make sense as normal human-language analogies.
Instead, they must be solved using vector arithmetic in the GloVe embedding
space.


Core Idea
---------

For:

    A : B :: C : ?

calculate:

    target = vector(C) + vector(B) - vector(A)

Then find the vocabulary token whose embedding is closest to the target
vector.

In Python:

    target = model[c] + (model[b] - model[a])
    result = model.similar_by_vector(target, topn=1)[0]


Files
-----

    chal.txt
        Original challenge input containing the 84 analogies.

    solve.py
        Python solver using glove-twitter-25.

    like-a-glove-writeup.md
        Detailed write-up explaining the challenge and solution.


Requirements
------------

Python 3

Install dependencies with:

    pip install gensim numpy scipy


Usage
-----

Place chal.txt and solve.py in the same directory.

Run:

    python3 solve.py

On the first run, Gensim will download the glove-twitter-25 model.

The script will:

    1. Parse every analogy.
    2. Calculate C + B - A.
    3. Find the closest GloVe token.
    4. Concatenate all recovered tokens.
    5. Normalize Unicode using NFKC.
    6. Print the final ASCII flag.


Unicode Trap
------------

The raw recovered result contains full-width Unicode characters such as:

    ４
    ０
    １
    ５

These look similar to normal ASCII digits but are different Unicode
characters.

The challenge says that the flag should be fully ASCII, so the solver uses:

    unicodedata.normalize("NFKC", raw)

Example:

    ４ -> 4
    ０ -> 0
    １ -> 1


Recovered Flag
--------------

    htb{h4rm0n1ou5_hymn_0f_h1ghd1m3ns10n4l_subl1me_5ymph0ny_0f_num3r1cal_nuanc3_1n_tr3mend0u5_t4p3stry_0f_t3xtu4l_7r4n5f0rma7ion}


Main Concepts Learned
---------------------

- Word embeddings
- GloVe
- Vector arithmetic
- Word analogies
- Nearest-neighbor similarity
- Unicode normalization
- The difference between human meaning and embedding-space relationships


Educational Note
----------------

A simple way to understand the challenge is:

    Words -> Numbers -> Vector Math -> Closest Word -> Flag

The important formula is:

    C + B - A


Author Notes
------------

This repository is intended as a learning write-up for CTF / AI-security
practice.

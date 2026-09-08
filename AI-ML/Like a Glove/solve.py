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
            r"Like (.+?) is to (.+?), (.+?) is to\\?",
            line
        )

        if not match:
            print(f"[-] Could not parse line {{i}}: {{line}}")
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
                f"[{{i:02}}] "
                f"{{a}} : {{b}} :: {{c}} : {{result}} "
                f"({{similarity:.6f}})"
            )

        except KeyError as e:
            print(f"[-] Missing GloVe token on line {{i}}: {{e}}")

raw = "".join(flag_parts)

print("\\n[+] RAW:")
print(raw)

# Convert compatibility/full-width Unicode characters to ASCII
flag = unicodedata.normalize("NFKC", raw)

print("\\n[+] NORMALIZED:")
print(flag)

if flag.startswith("htb{{"):
    print("\\n[+] FLAG FOUND!")

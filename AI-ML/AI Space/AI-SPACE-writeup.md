# Hack The Box — AI SPACE Write-up

## Challenge

**Category:** AI / Machine Learning

### Scenario

> You are assigned the important mission of locating and identifying the infamous space hacker. Your investigation begins by analyzing the data patterns and breach points identified in the latest cyber-attacks. Use the provided coordinates of the last known signal origins to narrow down his potential hideouts. Utilize advanced tracking algorithms to follow the digital footprint left by the hacker.

The challenge provides:

```text
distance_matrix.npy
```

The goal is to recover the hidden message and obtain the flag.

---

## Initial Analysis

The provided file is a NumPy `.npy` file.

We can load it with:

```python
import numpy as np

D = np.load("distance_matrix.npy")

print(D.shape)
print(D)
```

The matrix has the shape:

```text
(1808, 1808)
```

This strongly suggests that it contains pairwise information for **1808 points**.

Further inspection shows that:

- the matrix is square,
- the diagonal values are `0`,
- the matrix is symmetric,
- the values behave like distances between points.

Therefore:

```text
D[i][j]
```

represents the distance between point `i` and point `j`.

In other words, the challenge gives us the distances between all hidden points, but not their original coordinates.

---

# Main Idea

If the pairwise distances between a set of points are known, we can attempt to reconstruct their positions.

A suitable technique for this is:

```text
Multidimensional Scaling
```

or:

```text
MDS
```

More specifically, we can use:

```text
Classical Multidimensional Scaling
```

The idea is:

```text
Distance Matrix
      |
      v
Recover relative point positions
      |
      v
2D coordinates
      |
      v
Plot the points
      |
      v
Hidden message
```

The recovered shape may be rotated, mirrored, or upside down because distances alone do not preserve orientation.

---

# What is MDS?

Multidimensional Scaling tries to reconstruct coordinates while preserving the distances between points.

Suppose we know:

```text
A is 3 units from B
A is 5 units from C
B is 4 units from C
```

but we do not know:

```text
A = (?, ?)
B = (?, ?)
C = (?, ?)
```

MDS can reconstruct a coordinate system that approximately preserves those distances.

For this challenge, we have:

```text
1808 points
```

and all pairwise distances between them.

The hidden points turn out to lie in a two-dimensional structure, making it possible to reconstruct them as an image.

---

# Classical MDS

Let the distance matrix be:

```text
D
```

First, square each distance:

```python
D2 = D ** 2
```

If there are `n` points, create the centering matrix:

```python
J = np.eye(n) - np.ones((n, n)) / n
```

Then calculate the Gram matrix:

```python
B = -0.5 * J @ D2 @ J
```

This converts the pairwise distances into an inner-product representation.

Next, perform eigenvalue decomposition:

```python
eigenvalues, eigenvectors = np.linalg.eigh(B)
```

The largest eigenvalues correspond to the most important geometric dimensions.

Because the hidden data is essentially two-dimensional, we only need the top two dimensions.

---

# Recovering the Coordinates

Sort the eigenvalues from largest to smallest:

```python
idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

Then reconstruct the coordinates using the two largest positive eigenvalues:

```python
coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
```

Now:

```python
coords[:, 0]
```

contains the reconstructed X coordinates, while:

```python
coords[:, 1]
```

contains the reconstructed Y coordinates.

So we have converted:

```text
1808 x 1808 distance information
```

into:

```text
1808 x 2 coordinates
```

---

# Plotting the Points

The points can now be visualized:

```python
import matplotlib.pyplot as plt

plt.scatter(
    coords[:, 0],
    coords[:, 1],
    s=5
)

plt.axis("equal")
plt.show()
```

The point cloud reveals readable text, although its orientation may initially be wrong.

This happens because the same set of pairwise distances is preserved if the entire drawing is:

- rotated,
- reflected,
- flipped,
- translated.

Therefore, MDS can recover the correct geometry without knowing which direction should be considered "up".

For this challenge, flipping the axes makes the message easy to read:

```python
x = -coords[:, 0]
y = -coords[:, 1]
```

---

# Complete Solver

Save the following as:

```text
solve.py
```

```python
import numpy as np
import matplotlib.pyplot as plt


# Load the distance matrix
D = np.load("distance_matrix.npy")

print(f"[+] Matrix shape: {D.shape}")

n = D.shape[0]


# Step 1: Square all pairwise distances
D2 = D ** 2


# Step 2: Build the centering matrix
J = np.eye(n) - np.ones((n, n)) / n


# Step 3: Convert distances to a Gram matrix
B = -0.5 * J @ D2 @ J


print("[+] Performing eigenvalue decomposition...")


# Step 4: Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(B)


# Step 5: Sort the eigenvalues from largest to smallest
idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


print("[+] Largest eigenvalues:")
print(eigenvalues[:5])


# Step 6: Recover the top two dimensions
coords = (
    eigenvectors[:, :2]
    * np.sqrt(eigenvalues[:2])
)


# Step 7: Correct orientation
x = -coords[:, 0]
y = -coords[:, 1]


# Step 8: Plot the reconstructed points
plt.figure(figsize=(20, 4))

plt.scatter(
    x,
    y,
    s=5
)

plt.axis("equal")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    "recovered.png",
    dpi=250,
    bbox_inches="tight"
)

plt.show()

print("[+] Saved result as recovered.png")
```

---

# Running the Solver

Install the required packages:

```bash
pip install numpy matplotlib
```

Make sure these files are in the same directory:

```text
distance_matrix.npy
solve.py
```

Then execute:

```bash
python3 solve.py
```

The script performs the following steps:

```text
Load distance_matrix.npy
        |
        v
Square the distance matrix
        |
        v
Apply double centering
        |
        v
Perform eigenvalue decomposition
        |
        v
Take the two strongest dimensions
        |
        v
Recover X/Y coordinates
        |
        v
Plot the points
        |
        v
Flip orientation
        |
        v
Read the hidden message
```

---

# Recovered Message

After reconstructing and displaying the point cloud, the points spell:

```text
HTB{d1st4nt_spac3}
```

---

# Flag

```text
HTB{d1st4nt_spac3}
```

---

# Why the Challenge Works

The important weakness is that the challenge exposes a complete pairwise distance matrix.

Although the original coordinates are removed, a full Euclidean distance matrix still contains enough geometric information to reconstruct the relative positions of the points.

The coordinates themselves are not unique.

For example, these transformations do not change pairwise distances:

```text
rotation
reflection
translation
```

That is why the reconstruction may initially appear mirrored or upside down.

However, the underlying shape remains intact.

Once plotted correctly, the hidden text becomes visible.

---

# AI / Machine Learning Concept

This challenge demonstrates the idea of:

```text
Dimensionality Reduction
```

and specifically:

```text
Multidimensional Scaling
```

We start with high-dimensional relational information:

```text
1808 distances per point
```

and recover a low-dimensional representation:

```text
X coordinate
Y coordinate
```

Conceptually:

```text
High-dimensional distance information
                |
                v
     Classical MDS
                |
                v
       2D embedding
                |
                v
       Visual pattern
```

This is related to many machine-learning techniques where complex data is projected into a smaller number of dimensions so that its structure becomes easier to understand or visualize.

---

# Key Takeaways

## 1. Recognize the Data Structure

A square, symmetric matrix with zero diagonal values is a strong hint that it may be a distance matrix.

---

## 2. Pairwise Distances Can Leak Geometry

Removing the original coordinates does not necessarily hide the shape if all pairwise distances are still available.

---

## 3. Classical MDS Can Reconstruct Coordinates

The main formula is:

```text
B = -1/2 * J * D^2 * J
```

where:

```text
D = distance matrix
J = centering matrix
```

Eigenvalue decomposition of `B` then recovers the coordinate dimensions.

---

## 4. Orientation is Ambiguous

MDS may return the same point cloud:

```text
rotated
mirrored
upside down
```

because all of those transformations preserve distances.

---

## 5. Visualization Matters

The flag is not directly encoded as text inside the matrix.

The matrix encodes the geometry of points that form text.

Reconstructing and plotting those points reveals the answer.

---

# Short Summary

The provided `.npy` file is a:

```text
1808 x 1808 Euclidean distance matrix
```

We use **Classical Multidimensional Scaling** to reconstruct the original two-dimensional coordinates.

The solve is:

```text
distance_matrix.npy
        |
        v
Identify pairwise distances
        |
        v
Classical MDS
        |
        v
Eigenvalue decomposition
        |
        v
Recover two dimensions
        |
        v
Plot 1808 points
        |
        v
Correct orientation
        |
        v
HTB{d1st4nt_spac3}
```

Final flag:

```text
HTB{d1st4nt_spac3}
```

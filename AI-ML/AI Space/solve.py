import numpy as np
import matplotlib.pyplot as plt

# Load distance matrix
D = np.load("distance_matrix.npy")

print(f"[+] Matrix shape: {D.shape}")

# Number of points
n = D.shape[0]

# Square the distances
D2 = D ** 2

# Centering matrix
J = np.eye(n) - np.ones((n, n)) / n

# Classical MDS Gram matrix
B = -0.5 * J @ D2 @ J

print("[+] Performing eigenvalue decomposition...")

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(B)

# Sort from largest to smallest
idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("[+] Largest eigenvalues:")
print(eigenvalues[:5])

# Recover two-dimensional coordinates
coords = (
    eigenvectors[:, :2]
    * np.sqrt(eigenvalues[:2])
)

# MDS orientation is arbitrary.
# Flip both axes so the message is readable.
x = -coords[:, 0]
y = -coords[:, 1]

# Plot
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

print("[+] Saved reconstructed message as recovered.png")

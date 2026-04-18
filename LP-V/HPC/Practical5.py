#!/usr/bin/env python3
"""
Practical5.py

Student Performance Analytics using K-Means Clustering
Provides sequential and multiprocessing-parallel implementations.

Usage (default):
  python Practical5.py           # generate dataset, run seq+parallel
  python Practical5.py --generate --n 10000
  python Practical5.py --run --data data_students.csv --init init_centroids.csv

Requirements:
  pip install numpy

This script saves `data_students.csv` and `init_centroids.csv` in the current directory.
"""
import argparse
import time
import os
import math
import numpy as np
from multiprocessing import Pool, cpu_count


def generate_dataset(path_data, path_init, n=10000, d=3, k=3, seed=42):
    rng = np.random.default_rng(seed)
    # Create mixture of 3 performance clusters (Low, Average, High)
    proportions = np.array([0.35, 0.4, 0.25])
    proportions = (proportions / proportions.sum())
    means = [40, 65, 85]
    std = 8.0
    data_parts = []
    for i in range(k):
        ni = int(round(proportions[i] * n))
        # each subject is centered at the same mean for simplicity
        part = rng.normal(loc=means[i], scale=std, size=(ni, d))
        part = np.clip(part, 0, 100)
        data_parts.append(part)
    data = np.vstack(data_parts)
    # Shuffle
    rng.shuffle(data)
    np.savetxt(path_data, data, delimiter=',')
    # Choose deterministic initial centroids (by RNG seed)
    rng2 = np.random.default_rng(12345)
    if data.shape[0] < k:
        raise ValueError("n must be >= k")
    idx = rng2.choice(data.shape[0], size=k, replace=False)
    init = data[idx]
    np.savetxt(path_init, init, delimiter=',')
    print(f"Generated {data.shape[0]} samples ({d} features). Saved to {path_data}")
    print(f"Saved initial centroids to {path_init}")
    return data, init


def load_csv(path):
    return np.loadtxt(path, delimiter=',')


def inertia_for_labels(data, centroids, labels):
    return np.sum((data - centroids[labels])**2)


def kmeans_sequential(data, init_centroids, max_iter=100, tol=1e-4):
    t0 = time.perf_counter()
    centroids = init_centroids.copy()
    n, d = data.shape
    k = centroids.shape[0]
    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        # distances: n x k
        dists = np.sum((data[:, None, :] - centroids[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = centroids.copy()
        for j in range(k):
            pts = data[labels == j]
            if pts.shape[0] > 0:
                new_centroids[j] = pts.mean(axis=0)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift <= tol:
            break
    elapsed = time.perf_counter() - t0
    inertia = inertia_for_labels(data, centroids, labels)
    return centroids, labels, inertia, elapsed


def _partial_worker(args):
    chunk, centroids = args
    k, d = centroids.shape[0], centroids.shape[1]
    # distances: m x k
    dists = np.sum((chunk[:, None, :] - centroids[None, :, :])**2, axis=2)
    labels = np.argmin(dists, axis=1)
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros((k, d))
    for dim in range(d):
        sums[:, dim] = np.bincount(labels, weights=chunk[:, dim], minlength=k)
    inertia = np.sum((chunk - centroids[labels])**2)
    return sums, counts, inertia


def kmeans_parallel_mp(data, init_centroids, max_iter=100, tol=1e-4, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    t0 = time.perf_counter()
    centroids = init_centroids.copy()
    n, d = data.shape
    k = centroids.shape[0]
    labels = np.zeros(n, dtype=int)
    pool = Pool(processes=n_jobs)
    try:
        for it in range(max_iter):
            chunks = np.array_split(data, n_jobs)
            args = [(chunk, centroids) for chunk in chunks if chunk.shape[0] > 0]
            results = pool.map(_partial_worker, args)
            global_sums = np.zeros((k, d))
            global_counts = np.zeros(k, dtype=int)
            total_inertia = 0.0
            for sums, counts, inertia in results:
                global_sums += sums
                global_counts += counts
                total_inertia += inertia
            new_centroids = centroids.copy()
            for j in range(k):
                if global_counts[j] > 0:
                    new_centroids[j] = global_sums[j] / global_counts[j]
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= tol:
                break
        # final labeling (single-threaded for simplicity)
        dists = np.sum((data[:, None, :] - centroids[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = inertia_for_labels(data, centroids, labels)
    finally:
        pool.close()
        pool.join()
    elapsed = time.perf_counter() - t0
    return centroids, labels, inertia, elapsed


def centroid_similarity(a, b):
    # L2 norm between centroid sets (assumes same ordering)
    return np.linalg.norm(a - b)


def interpret_clusters(centroids):
    # Simple mapping based on centroid mean (small -> Low, mid -> Average, high -> High)
    means = centroids.mean(axis=1)
    order = np.argsort(means)
    labels = ['Low', 'Average', 'High']
    mapping = {}
    for i, idx in enumerate(order):
        if i < len(labels):
            mapping[idx] = labels[i]
        else:
            mapping[idx] = f'Cluster-{i}'
    return mapping


def run_all(path_data, path_init, n=10000, max_iter=100, tol=1e-4, n_jobs=None):
    data = load_csv(path_data)
    init = load_csv(path_init)
    print(f"Data shape: {data.shape}, k={init.shape[0]}")
    print("Running sequential K-Means (Python)...")
    c_seq, labels_seq, inertia_seq, t_seq = kmeans_sequential(data, init, max_iter=max_iter, tol=tol)
    print(f"  Seq time: {t_seq:.4f}s, inertia: {inertia_seq:.4f}")
    print("Running parallel K-Means (multiprocessing)...")
    c_par, labels_par, inertia_par, t_par = kmeans_parallel_mp(data, init, max_iter=max_iter, tol=tol, n_jobs=n_jobs)
    print(f"  Par time: {t_par:.4f}s, inertia: {inertia_par:.4f}")
    speedup = t_seq / t_par if t_par > 0 else float('inf')
    csim = centroid_similarity(c_seq, c_par)
    print(f"Speedup (seq/par): {speedup:.3f}")
    print(f"Centroid L2 difference between seq & par: {csim:.6f}")
    # Interpret clusters
    interp = interpret_clusters(c_seq)
    print("Cluster interpretations (based on centroid means):")
    for idx, lab in interp.items():
        print(f"  cluster {idx} -> {lab}, centroid mean={c_seq[idx].mean():.2f}")
    # Show sizes
    unique, counts = np.unique(labels_par, return_counts=True)
    print("Cluster sizes (parallel):")
    for u, c in zip(unique, counts):
        print(f"  cluster {u}: {c} students")
    return {
        'seq_time': t_seq, 'par_time': t_par, 'speedup': speedup,
        'inertia_seq': inertia_seq, 'inertia_par': inertia_par,
        'centroid_diff': csim
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generate', action='store_true', help='Generate dataset and initial centroids')
    ap.add_argument('--run', action='store_true', help='Run algorithms on provided data')
    ap.add_argument('--data', default='data_students.csv', help='Path to data CSV')
    ap.add_argument('--init', default='init_centroids.csv', help='Path to initial centroids CSV')
    ap.add_argument('--n', type=int, default=10000, help='Number of samples to generate')
    ap.add_argument('--d', type=int, default=3, help='Number of features (subjects)')
    ap.add_argument('--k', type=int, default=3, help='Number of clusters')
    ap.add_argument('--max-iter', type=int, default=100, help='Max iterations')
    ap.add_argument('--jobs', type=int, default=None, help='Number of worker processes')
    args = ap.parse_args()
    data_path = args.data
    init_path = args.init
    if args.generate:
        generate_dataset(data_path, init_path, n=args.n, d=args.d, k=args.k)
    if args.run or not (args.generate or args.run):
        # If nothing specified, generate then run
        if not os.path.exists(data_path) or not os.path.exists(init_path):
            print('Data or init not found, generating with defaults...')
            generate_dataset(data_path, init_path, n=args.n, d=args.d, k=args.k)
        run_all(data_path, init_path, n=args.n, max_iter=args.max_iter, n_jobs=args.jobs)


if __name__ == '__main__':
    main()

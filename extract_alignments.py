import argparse
import json
import os

import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm


def read(file, dtype='float', count=None):
    header = file.readline().split(' ')
    count = int(header[0]) if count == None else count
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    return (words, matrix)


def get_similarity(X, Y):
    return (cosine_similarity(X, Y) + 1.0) / 2.0


def get_mutual_sim(sim, sim_size):
    forward = np.eye(sim_size)[np.ravel(sim.argmax(axis=1))]
    backward = np.eye(sim_size)[np.ravel(sim.argmax(axis=0))]
    inter = forward * backward.transpose()
    return inter


def topk_mean(m, k, inplace=False):
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def equivalent_subw(l1_vocab, l2_vocab, sim, mutual_sim):
    alignment_dict = {}
    alignment_scores = {}
    for i in range(mutual_sim.shape[0]):
        aligned = np.max(mutual_sim[i])
        if aligned:
            alignment = np.argmax(mutual_sim[i])
            alignment_dict[l1_vocab[i]] = l2_vocab[alignment]
            alignment_scores[(l1_vocab[i], l2_vocab[alignment])] = np.mean(
                np.array([sim[i, alignment], sim[alignment, i]]))
    #             print(l1_vocab[i], alignment_dict[l1_vocab[i]], sim[i,alignment])
    return alignment_dict, alignment_scores


def forward(z):
    """forward pass for sparsemax
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    return np.maximum(0, z - tau_z)


def main(args):
    srcfile = open(args.src_emb, encoding='utf-8', errors='surrogateescape')
    eng_words, xw = read(srcfile)
    tgtfile = open(args.tgt_emb, encoding='utf-8', errors='surrogateescape')
    for_words, zw = read(tgtfile)

    sim_size = min(xw.shape[0], zw.shape[0])
    if args.similarity == "cosine":
        sim = get_similarity(xw, zw[:sim_size])

        real_msim = get_mutual_sim(sim, sim_size)
    elif args.similarity == "csls":
        sim = xw.dot(zw[:sim_size].T)
        knn_sim_fwd = topk_mean(sim, k=10)
        knn_sim_bwd = topk_mean(sim.T, k=10)
        sim -= knn_sim_fwd[:, np.newaxis] / 2 + knn_sim_bwd / 2

        real_msim = get_mutual_sim(sim, sim_size)
    elif args.similarity == "surface_form":
        joint_size = max(len(eng_words), len(for_words))
        real_msim = np.zeros((joint_size, joint_size))
        for i, eng_word in enumerate(eng_words):
            for j, for_word in enumerate(for_words):
                if eng_word == for_word:
                    real_msim[i, j] = 1
    else:
        raise ValueError("Not a valid similarity name.")

    print("Number of aligned subwords: {}".format(real_msim.sum()))

    alignment_dict, alignment_scores = equivalent_subw(eng_words, for_words, get_similarity(xw, zw[:sim_size]),
                                                       real_msim)
    print("Average similarity score: {}".format(np.mean(list(alignment_scores.values()))))

    output_dir = os.path.join("alignments", args.alignment_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "alignment_dict.json"), 'w') as fp:
        json.dump(alignment_dict, fp)

    if args.initialize:
        # we normalize before dot product so it is essentially cosine similarity
        xw_norm = normalize(xw, norm='l2', axis=1)
        zw_norm = normalize(zw, norm='l2', axis=1)
        # we change the similarity because we want target (sub)words as a weighted sum of the source (sub)words
        sim = zw_norm[:sim_size].dot(xw_norm[:sim_size].T)

        sparsemax = forward(sim)

        print("Calculating initialization weights for the weighted sum")
        probs = {}
        for i, tt in tqdm(enumerate(for_words[:sim_size]), total=sim_size):
            probs[tt] = {}
            ix = np.nonzero(sparsemax[i])[0]
            px = sparsemax[i][ix].tolist()
            wx = [eng_words[j] for j in ix.tolist()]
            probs[tt] = {w: p for w, p in zip(wx, px)}

        print("Saving initialization weights")
        torch.save(probs, os.path.join(output_dir, 'prob_vector'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('extract subword alignments')
    parser.add_argument('--src_emb', default='data/mono/txt/en/WP/mapped_en_el_embs.txt',
                        help='source (english) embeddings')
    parser.add_argument('--tgt_emb', default='data/mono/txt/el/WP/mapped_el_embs.txt',
                        help='target (foreign) language')
    parser.add_argument('--similarity', default='cosine', help='similarity metric')
    parser.add_argument('--alignment_dir', default='en-el',
                        help='where to store the alignment and the prob vector files')
    parser.add_argument('--initialize', action='store_true', help='whether to initialize the non-aligned subwords')
    args = parser.parse_args()
    main(args)

#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import numpy as np
import time
import os
import sys
import json
from datetime import datetime

# Add faiss module path
sys.path.insert(0, 'faiss/build/faiss/python')
sys.path.insert(0, 'faiss/benchs')

import faiss
import re

from multiprocessing.pool import ThreadPool
from datasets import ivecs_read

####################################################################
# Results tracking
####################################################################

results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
    },
    "config": {},
    "preprocessing": {},
    "coarse_quantizer": {},
    "index_training": {},
    "index_add": {
        "total_time": 0,
        "progress": []  # List of {"percent": X, "time": Y, "vectors_added": Z}
    },
    "search": []  # List of results per nprobe
}

####################################################################
# Parse command line
####################################################################


def usage():
    print("""

Usage: bench_cpu_1bn.py dataset indextype [options]

dataset: set of vectors to operate on.
   Supported: SIFT1M, SIFT2M, ..., SIFT1000M or Deep1B

indextype: any index type supported by index_factory that runs on CPU.

    General options

-nocache           do not read or write intermediate files

    Add options

-abs N             split adds in blocks of no more than N vectors

    Search options

-noptables         do not use precomputed tables in IVFPQ.
-qbs N             split queries in blocks of no more than N vectors
-nnn N             search N neighbors for each query
-nprobe 4,16,64    try this number of probes
-knngraph          instead of the standard setup for the dataset,
                   compute a k-nn graph with nnn neighbors per element
-oI xx%d.npy       output the search result indices to this numpy file,
                   %d will be replaced with the nprobe
-oD xx%d.npy       output the search result distances to this file

""", file=sys.stderr)
    sys.exit(1)


# default values

dbname = None
index_key = None

# CPU-only mode - no GPU parameters
add_batch_size = 32768
query_batch_size = 16384
nprobes = [1 << l for l in range(9)]
knngraph = False
use_precomputed_tables = True
use_cache = True
nnn = 10
I_fname = None
D_fname = None

args = sys.argv[1:]

while args:
    a = args.pop(0)
    if a == '-h':
        usage()
    elif a == '-noptables':
        use_precomputed_tables = False
    elif a == '-abs':
        add_batch_size = int(args.pop(0))
    elif a == '-qbs':
        query_batch_size = int(args.pop(0))
    elif a == '-nnn':
        nnn = int(args.pop(0))
    elif a == '-nocache':
        use_cache = False
    elif a == '-knngraph':
        knngraph = True
    elif a == '-nprobe':
        nprobes = [int(x) for x in args.pop(0).split(',')]
    elif a == '-oI':
        I_fname = args.pop(0)
    elif a == '-oD':
        D_fname = args.pop(0)
    elif not dbname:
        dbname = a
    elif not index_key:
        index_key = a
    else:
        print("argument %s unknown" % a, file=sys.stderr)
        sys.exit(1)

cacheroot = '/tmp/bench_gpu_1bn'

if not os.path.isdir(cacheroot):
    print("%s does not exist, creating it" % cacheroot)
    os.mkdir(cacheroot)

# Create results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
if not os.path.isdir(results_dir):
    print("Creating results directory: %s" % results_dir)
    os.makedirs(results_dir)

# Store config in results
results["config"] = {
    "dbname": dbname,
    "index_key": index_key,
    "add_batch_size": add_batch_size,
    "query_batch_size": query_batch_size,
    "nprobes": nprobes,
    "use_precomputed_tables": use_precomputed_tables,
    "nnn": nnn,
    "mode": "cpu",
}

#################################################################
# Small Utility Functions
#################################################################

# we mem-map the biggest files to avoid having them in memory all at
# once


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x


def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))


def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    def prepare_block(i01):
        i0, i1 = i01
        xb = sanitize(x[i0:i1])
        return i0, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)


def eval_intersection_measure(gt_I, I):
    """ measure intersection measure (used for knngraph)"""
    inter = 0
    rank = I.shape[1]
    assert gt_I.shape[1] >= rank
    for q in range(nq_gt):
        inter += faiss.ranklist_intersection_size(
            rank, faiss.swig_ptr(gt_I[q, :]),
            rank, faiss.swig_ptr(I[q, :].astype('int64')))
    return inter / float(rank * nq_gt)


#################################################################
# Prepare dataset
#################################################################

print("Preparing dataset", dbname)

if dbname.startswith('SIFT'):
    # SIFT1M to SIFT1000M
    dbsize = int(dbname[4:-1])
    xb = mmap_bvecs('faiss/benchs/bigann/bigann_base.bvecs')
    xq = mmap_bvecs('faiss/benchs/bigann/bigann_query.bvecs')
    xt = mmap_bvecs('faiss/benchs/bigann/bigann_learn.bvecs')

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]

    gt_I = ivecs_read('faiss/benchs/bigann/gnd/idx_%dM.ivecs' % dbsize)

elif dbname == 'Deep1B':
    xb = mmap_fvecs('faiss/benchs/deep1b/base.fvecs')
    xq = mmap_fvecs('faiss/benchs/deep1b/deep1B_queries.fvecs')
    xt = mmap_fvecs('faiss/benchs/deep1b/learn.fvecs')
    # deep1B's train is is outrageously big
    xt = xt[:10 * 1000 * 1000]
    gt_I = ivecs_read('faiss/benchs/deep1b/deep1B_groundtruth.ivecs')

else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)


if knngraph:
    # convert to knn-graph dataset
    xq = xb
    xt = xb
    # we compute the ground-truth on this number of queries for validation
    nq_gt = 10000
    gt_sl = 100

    # ground truth will be computed below
    gt_I = None


print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape,
    gt_I.shape if gt_I is not None else None))


#################################################################
# Parse index_key and set cache files
#
# The index_key is a valid factory key that would work, but we
# decompose the training to do it faster
#################################################################


pat = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                 '(IVF[0-9]+),' +
                 '(PQ[0-9]+|Flat)')

matchobject = pat.match(index_key)

assert matchobject, 'could not parse ' + index_key

mog = matchobject.groups()

preproc_str = mog[0]
ivf_str = mog[2]
pqflat_str = mog[3]

ncent = int(ivf_str[3:])

prefix = ''

if knngraph:
    gt_cachefile = '%s/BK_gt_%s.npy' % (cacheroot, dbname)
    prefix = 'BK_'
    # files must be kept distinct because the training set is not the
    # same for the knngraph

if preproc_str:
    preproc_cachefile = '%s/%spreproc_%s_%s.vectrans' % (
        cacheroot, prefix, dbname, preproc_str[:-1])
else:
    preproc_cachefile = None
    preproc_str = ''

cent_cachefile = '%s/%scent_%s_%s%s.npy' % (
    cacheroot, prefix, dbname, preproc_str, ivf_str)

index_cachefile = '%s/%s%s_%s%s,%s.index' % (
    cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)


if not use_cache:
    preproc_cachefile = None
    cent_cachefile = None
    index_cachefile = None

print("cachefiles:")
print(preproc_cachefile)
print(cent_cachefile)
print(index_cachefile)


#################################################################
# CPU mode - no GPU resources needed
#################################################################

print("CPU mode - using system RAM for indexing and search")


#################################################################
# Prepare ground truth (for the knngraph)
#################################################################


def compute_GT():
    print("compute GT")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])

    db_gt = faiss.IndexFlatL2(d)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt.add(xsl)
        D, I = db_gt.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt.reset()
        print("\r   %d/%d, %.3f s" % (i0, n, time.time() - t0), end=' ')
    print()
    heaps.reorder()

    print("GT time: %.3f s" % (time.time() - t0))
    return gt_I


if knngraph:

    if gt_cachefile and os.path.exists(gt_cachefile):
        print("load GT", gt_cachefile)
        gt_I = np.load(gt_cachefile)
    else:
        gt_I = compute_GT()
        if gt_cachefile:
            print("store GT", gt_cachefile)
            np.save(gt_cachefile, gt_I)

#################################################################
# Prepare the vector transformation object (pure CPU)
#################################################################


def train_preprocessor():
    print("train preproc", preproc_str)
    d = xt.shape[1]
    t0 = time.time()
    if preproc_str.startswith('OPQ'):
        fi = preproc_str[3:-1].split('_')
        m = int(fi[0])
        dout = int(fi[1]) if len(fi) == 2 else d
        preproc = faiss.OPQMatrix(d, m, dout)
    elif preproc_str.startswith('PCAR'):
        dout = int(preproc_str[4:-1])
        preproc = faiss.PCAMatrix(d, dout, 0, True)
    else:
        assert False
    preproc.train(sanitize(xt[:1000000]))
    train_time = time.time() - t0
    print("preproc train done in %.3f s" % train_time)
    results["preprocessing"] = {
        "type": preproc_str,
        "train_time": train_time,
    }
    return preproc


def get_preprocessor():
    if preproc_str:
        if not preproc_cachefile or not os.path.exists(preproc_cachefile):
            preproc = train_preprocessor()
            if preproc_cachefile:
                print("store", preproc_cachefile)
                faiss.write_VectorTransform(preproc, preproc_cachefile)
        else:
            print("load", preproc_cachefile)
            preproc = faiss.read_VectorTransform(preproc_cachefile)
    else:
        d = xb.shape[1]
        preproc = IdentPreproc(d)
    return preproc


#################################################################
# Prepare the coarse quantizer
#################################################################


def train_coarse_quantizer(x, k, preproc):
    d = preproc.d_out
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    # clus.niter = 2
    clus.max_points_per_centroid = 10000000

    print("apply preproc on shape", x.shape, 'k=', k)
    t0 = time.time()
    x = preproc.apply_py(sanitize(x))
    preproc_time = time.time() - t0
    print("   preproc %.3f s output shape %s" % (preproc_time, x.shape))

    # Use CPU index for training
    index = faiss.IndexFlatL2(d)

    t1 = time.time()
    clus.train(x, index)
    cluster_time = time.time() - t1
    centroids = faiss.vector_float_to_array(clus.centroids)

    results["coarse_quantizer"] = {
        "num_centroids": k,
        "preproc_time": preproc_time,
        "cluster_time": cluster_time,
        "total_time": preproc_time + cluster_time,
    }

    return centroids.reshape(k, d)


def prepare_coarse_quantizer(preproc):

    if cent_cachefile and os.path.exists(cent_cachefile):
        print("load centroids", cent_cachefile)
        centroids = np.load(cent_cachefile)
    else:
        nt = max(1000000, 256 * ncent)
        print("train coarse quantizer...")
        t0 = time.time()
        centroids = train_coarse_quantizer(xt[:nt], ncent, preproc)
        print("Coarse train time: %.3f s" % (time.time() - t0))
        if cent_cachefile:
            print("store centroids", cent_cachefile)
            np.save(cent_cachefile, centroids)

    coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
    coarse_quantizer.add(centroids)

    return coarse_quantizer


#################################################################
# Make index and add elements to it
#################################################################


def prepare_trained_index(preproc):

    coarse_quantizer = prepare_coarse_quantizer(preproc)
    d = preproc.d_out
    if pqflat_str == 'Flat':
        print("making an IVFFlat index")
        idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, ncent,
                                       faiss.METRIC_L2)
    else:
        m = int(pqflat_str[2:])
        print("making an IVFPQ index, m = ", m)
        idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8)

    coarse_quantizer.this.disown()
    idx_model.own_fields = True

    # finish training on CPU
    t0 = time.time()
    print("Training vector codes")
    x = preproc.apply_py(sanitize(xt[:1000000]))
    idx_model.train(x)
    train_time = time.time() - t0
    print("  done %.3f s" % train_time)

    results["index_training"] = {
        "index_type": pqflat_str,
        "train_time": train_time,
    }

    return idx_model


def compute_populated_index(preproc):
    """Add elements to a CPU index."""

    index = prepare_trained_index(preproc)

    print("add...")
    t0 = time.time()
    nb = xb.shape[0]
    last_progress_pct = 0
    results["index_add"]["progress"] = []
    results["index_add"]["total_vectors"] = nb

    for i0, xs in dataset_iterator(xb, preproc, add_batch_size):
        i1 = i0 + xs.shape[0]
        index.add_with_ids(xs, np.arange(i0, i1))

        # Record progress every 5%
        current_pct = int((i1 / nb) * 100)
        if current_pct >= last_progress_pct + 5:
            elapsed = time.time() - t0
            results["index_add"]["progress"].append({
                "percent": current_pct,
                "time": elapsed,
                "vectors_added": i1,
            })
            last_progress_pct = (current_pct // 5) * 5

        print('\r%d/%d (%.3f s)  ' % (
            i0, nb, time.time() - t0), end=' ')
        sys.stdout.flush()

    total_add_time = time.time() - t0
    print("Add time: %.3f s" % total_add_time)

    # Record final progress
    results["index_add"]["progress"].append({
        "percent": 100,
        "time": total_add_time,
        "vectors_added": nb,
    })
    results["index_add"]["total_time"] = total_add_time

    return index


def get_populated_index(preproc):
    # No caching - always create fresh CPU index
    index = compute_populated_index(preproc)
    return index



#################################################################
# Perform search
#################################################################


def eval_dataset(index, preproc):

    nq_gt = gt_I.shape[0]
    print("search...")
    sl = query_batch_size
    nq = xq.shape[0]

    results["search"] = []

    for nprobe in nprobes:
        # Set nprobe directly for CPU index
        index.nprobe = nprobe
        t0 = time.time()

        search_result = {
            "nprobe": nprobe,
            "num_queries": nq,
        }

        if sl == 0:
            D, I = index.search(preproc.apply_py(sanitize(xq)), nnn)
        else:
            I = np.empty((nq, nnn), dtype='int32')
            D = np.empty((nq, nnn), dtype='float32')

            inter_res = ''

            for i0, xs in dataset_iterator(xq, preproc, sl):
                print('\r%d/%d (%.3f s%s)   ' % (
                    i0, nq, time.time() - t0, inter_res), end=' ')
                sys.stdout.flush()

                i1 = i0 + xs.shape[0]
                Di, Ii = index.search(xs, nnn)

                I[i0:i1] = Ii
                D[i0:i1] = Di

                if knngraph and not inter_res and i1 >= nq_gt:
                    ires = eval_intersection_measure(
                        gt_I[:, :nnn], I[:nq_gt])
                    inter_res = ', %.4f' % ires

        t1 = time.time()
        search_time = t1 - t0
        search_result["search_time"] = search_time
        search_result["qps"] = nq / search_time  # queries per second

        if knngraph:
            ires = eval_intersection_measure(gt_I[:, :nnn], I[:nq_gt])
            print("  probe=%-3d: %.3f s rank-%d intersection results: %.4f" % (
                nprobe, search_time, nnn, ires))
            search_result["intersection_measure"] = ires
        else:
            print("  probe=%-3d: %.3f s" % (nprobe, search_time), end=' ')
            gtc = gt_I[:, :1]
            nq = xq.shape[0]
            recall_results = {}
            for rank in (1, 10, 100):
                if rank > nnn:
                    continue
                nok = (I[:, :rank] == gtc).sum()
                recall = nok / float(nq)
                print("1-R@%d: %.4f" % (rank, recall), end=' ')
                recall_results["1-R@%d" % rank] = recall
            print()
            search_result["recall"] = recall_results

        results["search"].append(search_result)

        if I_fname:
            I_fname_i = I_fname % I
            print("storing", I_fname_i)
            np.save(I, I_fname_i)
        if D_fname:
            D_fname_i = I_fname % I
            print("storing", D_fname_i)
            np.save(D, D_fname_i)


#################################################################
# Driver
#################################################################


preproc = get_preprocessor()

index = get_populated_index(preproc)

eval_dataset(index, preproc)

# make sure index is deleted before the resources
del index

#################################################################
# Save results to JSON
#################################################################

# Generate result filename with config
result_filename = f"{dbname}_{index_key.replace(',', '_')}_cpu.json"
result_filepath = os.path.join(results_dir, result_filename)

# Calculate summary statistics
results["summary"] = {
    "total_build_time": (
        results.get("preprocessing", {}).get("train_time", 0) +
        results.get("coarse_quantizer", {}).get("total_time", 0) +
        results.get("index_training", {}).get("train_time", 0) +
        results.get("index_add", {}).get("total_time", 0)
    ),
    "best_search": None,
}

# Find best search result (highest recall at rank 1 or best intersection measure)
if results["search"]:
    if knngraph:
        best = max(results["search"], key=lambda x: x.get("intersection_measure", 0))
        results["summary"]["best_search"] = {
            "nprobe": best["nprobe"],
            "intersection_measure": best.get("intersection_measure"),
            "qps": best["qps"],
        }
    else:
        best = max(results["search"], key=lambda x: x.get("recall", {}).get("1-R@1", 0))
        results["summary"]["best_search"] = {
            "nprobe": best["nprobe"],
            "recall_1R1": best.get("recall", {}).get("1-R@1"),
            "qps": best["qps"],
        }

# Save results
with open(result_filepath, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved to: %s" % result_filepath)
print("="*60)

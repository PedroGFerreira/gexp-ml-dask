import gc
import csv
import os.path
import time
import timeit

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from dask.distributed import Client
from memory_profiler import memory_usage

from functions.dask_pipeline import dask_pipeline
from functions.nondask_pipeline import nondask_pipeline

if __name__ == '__main__':
    xgb.set_config(verbosity=0)

    fps_dask = [('data/brca_undersample/parquet/n1205_f10000_brca_fpkm.parquet',
                 'data/brca_undersample/parquet/n1205_brca_subtypes.parquet'),
                ('data/brca_undersample/parquet/n1205_f20000_brca_fpkm.parquet',
                 'data/brca_undersample/parquet/n1205_brca_subtypes.parquet'),
                ('data/brca_undersample/parquet/n1205_f40000_brca_fpkm.parquet',
                 'data/brca_undersample/parquet/n1205_brca_subtypes.parquet')
                ]

    fps_nondask = [('data/brca_undersample/csv/n1205_f10000_brca_fpkm.csv',
                    'data/brca_undersample/csv/n1205_brca_subtypes.csv'),
                   ('data/brca_undersample/csv/n1205_f20000_brca_fpkm.csv',
                    'data/brca_undersample/csv/n1205_brca_subtypes.csv'),
                   ('data/brca_undersample/csv/n1205_f40000_brca_fpkm.csv',
                    'data/brca_undersample/csv/n1205_brca_subtypes.csv')
                   ]

    peak_mem = {}
    runtime = {}
    cv_scores = {}
    eval_scores = {}

    for fp in fps_dask:
        threaded_client = Client(n_workers=1)
        print(threaded_client)

        dimensions = f"{os.path.basename(fp[0]).split('_')[1]}"
        print(dimensions)

        dask_threaded_memory, retval = memory_usage([dask_pipeline, (fp,)], max_usage=True, include_children=True,
                                                    retval=True)
        mean_cv_score, std_cv_score, eval_score = retval

        dask_threaded_runtime = min(timeit.repeat("dask_pipeline(fp)",
                                                  setup="from __main__ import dask_pipeline, fp", repeat=3, number=1))
        print(dask_threaded_memory)
        print(dask_threaded_runtime)

        peak_mem[dimensions] = {'Dask (Threaded)': dask_threaded_memory}
        runtime[dimensions] = {'Dask (Threaded)': dask_threaded_runtime}
        cv_scores[dimensions] = {'Dask (Threaded)': (mean_cv_score, std_cv_score)}
        eval_scores[dimensions] = {'Dask (Threaded)': eval_score}

        threaded_client.shutdown()
        gc.collect()
        time.sleep(60)

    print(peak_mem)
    print(runtime)
    print(cv_scores)
    print(eval_scores)

    for fp in fps_dask:
        distributed_client = Client()
        print(distributed_client)

        dimensions = f"{os.path.basename(fp[0]).split('_')[1]}"
        print(dimensions)

        dask_distributed_memory, retval = memory_usage([dask_pipeline, (fp,)], max_usage=True, include_children=True,
                                                       retval=True)
        mean_cv_score, std_cv_score, eval_score = retval

        dask_distributed_runtime = min(timeit.repeat("dask_pipeline(fp)",
                                                     setup="from __main__ import dask_pipeline, fp", repeat=3,
                                                     number=1))

        print(dask_distributed_memory)
        print(dask_distributed_runtime)

        peak_mem[dimensions].update({'Dask (Distributed)': dask_distributed_memory})
        runtime[dimensions].update({'Dask (Distributed)': dask_distributed_runtime})
        cv_scores[dimensions].update({'Dask (Distributed)': (mean_cv_score, std_cv_score)})
        eval_scores[dimensions].update({'Dask (Distributed)': eval_score})

        distributed_client.shutdown()
        gc.collect()
        time.sleep(60)

    print(peak_mem)
    print(runtime)
    print(cv_scores)
    print(eval_scores)

    for fp in fps_nondask:
        dimensions = f"{os.path.basename(fp[0]).split('_')[1]}"
        print(dimensions)

        nondask_memory, retval = memory_usage([nondask_pipeline, (fp,)], max_usage=True, include_children=True,
                                              retval=True)
        mean_cv_score, std_cv_score, eval_score = retval

        nondask_runtime = min(timeit.repeat("nondask_pipeline(fp)", setup="from __main__ import nondask_pipeline, fp",
                                            repeat=3, number=1))

        peak_mem[dimensions].update({'Scientific Python Ecosystem': nondask_memory})
        runtime[dimensions].update({'Scientific Python Ecosystem': nondask_runtime})
        cv_scores[dimensions].update({'Scientific Python Ecosystem': (mean_cv_score, std_cv_score)})
        eval_scores[dimensions].update({'Scientific Python Ecosystem': eval_score})
        gc.collect()

    print(peak_mem)
    print(runtime)
    print(cv_scores)
    print(eval_scores)

    with open("benchmark_results/featurewise/brca_undersample_peakmem_benchmark_featurewise.csv", "w") as outfile:
        w_mem = csv.writer(outfile)
        w_mem.writerow(['f Features', 'Framework', "Peak Memory (MiB)"])
        for key, val in peak_mem.items():
            for subkey, subval in val.items():
                w_mem.writerow([key, subkey, subval])

    with open("benchmark_results/featurewise/brca_undersample_runtime_benchmark_featurewise.csv", "w") as outfile:
        w_time = csv.writer(outfile)
        w_time.writerow(['f Features', 'Framework', "Fastest Runtime (s)"])
        for key, val in runtime.items():
            for subkey, subval in val.items():
                w_time.writerow([key, subkey, subval])

    with open("benchmark_results/featurewise/brca_undersample_evalscore_benchmark_featurewise.csv", "w") as outfile:
        w_eval = csv.writer(outfile)
        w_eval.writerow(['f Features', 'Framework', "Evaluation Score (Accuracy)"])
        for key, val in eval_scores.items():
            for subkey, subval in val.items():
                w_eval.writerow([key, subkey, subval])

    pd.DataFrame(runtime).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"])
    plt.title('Minimum runtime for different machine learning frameworks')
    plt.xticks(rotation=360)
    plt.ylabel('Minimum Runtime (s)')
    plt.xlabel('BRCA Undersample With f Features (n samples = 1205)')
    plt.savefig("benchmark_results/featurewise/brca_undersample_runtime_benchmark_featurewise.pdf")

    pd.DataFrame(peak_mem).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"])
    plt.title('Peak memory consumption for different machine learning frameworks')
    plt.xticks(rotation=360)
    plt.ylabel('Peak Memory Consumption (MiB)')
    plt.xlabel('BRCA Undersample With f Features (n samples = 1205)')
    plt.savefig("benchmark_results/featurewise/brca_undersample_peakmem_benchmark_featurewise.pdf")

    pd.DataFrame(eval_scores).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"])
    plt.title('Evaluation accuracy scores for different machine learning frameworks')
    plt.xticks(rotation=360)
    plt.ylabel('Accuracy')
    plt.xlabel('BRCA Undersample With f Features (n samples = 1205)')
    plt.savefig("benchmark_results/featurewise/brca_undersample_evalscore_benchmark_featurewise.pdf")

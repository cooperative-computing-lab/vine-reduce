# /usr/bin/env python
"""Runs a coffea processor over a ttbar-EFT-style analysis via vine_reduce's
TaskVine executor, against a real cluster.

Unlike examples/quick_start and examples/cortado, this script does not spawn
its own local workers with vine.Factory: it opens a vine.Manager on --port
and waits for independently-launched vine_worker processes (or a batch-system
submission) to connect, the normal way TaskVine is used on a real cluster
(see TaskVineDistributor's "Manager-only, external workers" note in
PLAN.md). There is no local/no-cluster fallback in this script - for that,
run analysis_processor.py directly through coffea's own IterativeExecutor
(see the README in this directory) instead of through vine_reduce.

Preprocessing (turning each input file into a num_entries count, so
vine_reduce knows where chunk boundaries fall) is done with coffea's own
dataset_tools.preprocess(), on the same vine.DaskVine manager - and hence the
same worker pool - as the actual processing/reduction tasks that follow, via
TaskVineDistributor's `manager=` constructor argument, rather than opening a
second manager/port for preprocessing alone.
"""

import argparse
import glob
import gzip
import importlib
import json
import os
import pprint
import shutil
import time

import cloudpickle
import ndcctools.taskvine as vine
import yaml
from coffea.dataset_tools import preprocess
from coffea.nanoevents import NanoAODSchema

from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor


def get_filename_from_path(filename):
    full_file_name = os.path.basename(filename)
    base, extension = os.path.splitext(full_file_name)

    return base


def load_json_to_samplesdict(inputFile, prefix):
    samplesdict = {}
    json_dict = read_json_file(inputFile)
    sample_name = get_filename_from_path(inputFile)
    samplesdict[sample_name] = json_dict
    samplesdict[sample_name]["redirector"] = prefix

    return samplesdict


def build_coffea_fileset(sample, treename):
    """Builds one entry of the `fileset` dict coffea's own
    dataset_tools.preprocess() expects: {"files": {url: {"object_path":
    treename}}, "metadata": {...everything but "files"...}}."""
    files_dict = {}
    metadata = dict(sample)
    del metadata["files"]
    for f in sample["files"]:
        fname = sample["redirector"] + f
        files_dict[fname] = {"object_path": treename}

    return {"files": files_dict, "metadata": metadata}


def read_json_file(filename):
    with open(filename) as f:
        return json.load(f)


def read_yaml_file(filename):
    with open(filename) as f:
        return yaml.safe_load(f)


def load_all_results(results_dir, dataset_names, processor_names):
    """Final results land under results_dir/<dataset>/<processor>/ as a
    single compressed, pickled file (name includes a random uuid, hence the
    glob) - vr.compute() itself returns nothing, so this is how the final
    histograms get back into memory afterwards."""
    hists = {}
    for dataset_name in dataset_names:
        hists[dataset_name] = {}
        for processor_name in processor_names:
            pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
            (result_file,) = glob.glob(pattern)
            hists[dataset_name][processor_name] = serialization.load(result_file)
    return hists


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    # TODO: make this an input argument with a default or make it based on --outname
    results_dir = f"/users/{os.environ['USER']}/vr_coffea_test/"

    parser = argparse.ArgumentParser(description="You can customize your run")
    parser.add_argument("inputFile", nargs="?", help="Json file(s) containing files and metadata")
    parser.add_argument(
        "--prefix",
        "-r",
        nargs="?",
        default="",
        help="Prefix or redirector to look for the files",
    )
    parser.add_argument(
        "--chunksize", "-s", default=100000, type=int, help="Number of events per chunk"
    )
    parser.add_argument(
        "--outname",
        "-o",
        default="histos",
        help="Name of the output file with histograms",
    )
    parser.add_argument("--treename", default="Events", help="Name of the tree inside the files")
    parser.add_argument(
        "--wc-list",
        action="extend",
        nargs="+",
        help="Specify a list of Wilson coefficients to use in filling histograms.",
    )
    parser.add_argument(
        "--hist-list",
        action="extend",
        nargs="+",
        help="Specify a list of histograms to fill.",
    )
    parser.add_argument(
        "--port",
        default="9123-9130",
        help="Specify the TaskVine port. An integer PORT or an integer range PORT_MIN-PORT_MAX.",
    )
    parser.add_argument(
        "--processor",
        "-p",
        default="analysis_processor.py",
        help="Specify processor file name",
    )

    args = parser.parse_args()
    inputFile = args.inputFile
    prefix = args.prefix
    chunksize = args.chunksize
    outname = args.outname
    treename = args.treename
    wc_lst = args.wc_list if args.wc_list is not None else []
    proc_file = args.processor
    proc_name = args.processor[:-3]
    hist_lst = args.hist_list
    ports = args.port

    print("\n\nrunning with processor: ", proc_file, "\n")

    analysis_processor = importlib.import_module(proc_name)

    # Check if input is json or yaml
    if inputFile.endswith(".json"):
        isJson = True
        isYaml = False
    elif (inputFile.endswith(".yaml")) or (inputFile.endswith(".yml")):
        isJson = False
        isYaml = True
    else:
        raise ValueError(
            f"Expects a .json, .yaml, or .yml for the input file. inputFile ={inputFile}"
        )

    # construct the taskvine port range
    port = list(map(int, ports.split("-")))
    if len(port) < 1:
        raise ValueError("At least one port value should be specified.")
    if len(port) > 2:
        raise ValueError("More than one port range was specified.")
    if len(port) == 1:
        # convert single values into a range of one element
        port.append(port[0])

    # Fill Samples Dictionary ###
    samplesdict = {}

    if isJson:
        samplesdict.update(load_json_to_samplesdict(inputFile, prefix))

    elif isYaml:
        yaml_dict = read_yaml_file(inputFile)
        if "jsonFiles" in yaml_dict.keys():
            redirector = yaml_dict["redirector"]
            jsonFiles = yaml_dict["jsonFiles"]

            for f in jsonFiles:
                samplesdict.update(load_json_to_samplesdict(f, redirector))
        else:
            for item in yaml_dict:
                redirector = yaml_dict[item]["redirector"]
                jsonFiles = yaml_dict[item]["jsonFiles"]

                for f in jsonFiles:
                    samplesdict.update(load_json_to_samplesdict(f, redirector))

    # Fill WC list ###
    if len(wc_lst) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]["WCnames"]:
                if wc not in wc_lst:
                    wc_lst.append(wc)
    if len(wc_lst) > 0:
        print(f"Wilson Coefficients: {wc_lst}")
    else:
        print("Wilson Coefficients: NONE SPECIFIED")

    print("\n\n samplesdict:")
    pprint.pprint(samplesdict)

    tstart = time.time()

    # One manager for both preprocessing and the actual processing/reduction
    # below: DaskVine is a vine.Manager that also implements dask's scheduler
    # interface (its .get method), so coffea's own dataset_tools.preprocess()
    # can run its (dask-based) file-opening work as TaskVine tasks against
    # the same worker pool that TaskVineDistributor submits to afterwards -
    # see TaskVineDistributor's manager= argument.
    mgr = vine.DaskVine(
        port=port,
        name=f"{os.environ['USER']}-vr-coffea",
    )
    mgr.tune("hungry-minimum", 1)

    # Check if the X509 proxy file exists
    x509_proxy = f"/tmp/x509up_u{os.getuid()}"
    if not os.path.exists(x509_proxy):
        print(f"Warning: X509 proxy file {x509_proxy} does not exist. Setting to None.")
        x509_proxy = None
    else:
        shutil.copy(x509_proxy, "./proxy.pem")

    fileset = {sname: build_coffea_fileset(samplesdict[sname], treename) for sname in samplesdict}

    print("\nPreprocessing data with TaskVine...")
    available, _ = preprocess(
        fileset,
        scheduler=mgr.get,
        skip_bad_files=True,
    )

    distributor = TaskVineDistributor(
        manager=mgr,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
    )

    extra_files = [proc_file, os.path.join(here, "channels.json")]
    environment_variables = {}
    if x509_proxy is not None:
        extra_files.append("proxy.pem")
        environment_variables["X509_USER_PROXY"] = "proxy.pem"

    vr = VineReduceCoffea(
        processors={
            "ee_chan": analysis_processor.AnalysisProcessor(
                samples=samplesdict,
                lep_cat="ee",
                wc_names_lst=wc_lst,
                hist_lst=hist_lst,
            ),
            "mm_chan": analysis_processor.AnalysisProcessor(
                samples=samplesdict,
                lep_cat="mm",
                wc_names_lst=wc_lst,
                hist_lst=hist_lst,
            ),
            "em_chan": analysis_processor.AnalysisProcessor(
                samples=samplesdict,
                lep_cat="em",
                wc_names_lst=wc_lst,
                hist_lst=hist_lst,
            ),
        },
        input=available,
        schema=NanoAODSchema,
        object_path=treename,
        chunksize=chunksize,
        extra_files=extra_files,
        environment_variables=environment_variables,
        results_dir=results_dir,
        distributor=distributor,
    )
    vr.compute()
    distributor.shutdown()

    hists = load_all_results(results_dir, list(available.keys()), list(vr.processors.keys()))
    print("\n\n resulting hists: ")
    pprint.pprint(hists)

    # Save Output ###
    outpath = "."
    out_pkl_file = os.path.join(outpath, f"{outname}.pkl.gz")
    print(f"\n\n Saving output to {out_pkl_file}")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(hists, fout)
        print("Done! \n\n")

    tend = time.time()
    print(f"\n\n Total processing time: {tend-tstart}")

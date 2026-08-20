# /usr/bin/env python
import argparse
import json
import yaml
import time
import cloudpickle
import gzip
import os
import shutil
import importlib
import pprint

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from vine_reduce import CoffeaVineReduce, preprocess

import ndcctools.taskvine as vine


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


def preprocessing_for_taskvine(samplesdict):
    flist = {}
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]["redirector"]
        flist[sname] = [(redirector + f) for f in samplesdict[sname]["files"]]

    return flist


def preprocessing_for_vr(sample):
    files_dict = {}
    metadata = dict(sample)
    del metadata["files"]
    for f in sample["files"]:
        fname = sample["redirector"] + f
        files_dict[fname] = {"object_path": "Events"}

    return {"files": files_dict, "metadata": metadata}


def read_json_file(filename):
    with open(filename) as f:
        return json.load(f)


def read_yaml_file(filename):
    with open(filename) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # TODO: make this an input argument with a default or make it based on --outname
    results_dir = f"/users/{os.environ['USER']}/vr_coffea_test/"

    # TODO: add IterativeExecutor Options
    known_executors = ["iterative", "vr"]

    parser = argparse.ArgumentParser(description="You can customize your run")
    parser.add_argument(
        "inputFile", nargs="?", help="Json file(s) containing files and metadata"
    )
    parser.add_argument("--executor", "-x", default="vr", help="Which executor to use")
    parser.add_argument(
        "--prefix",
        "-r",
        nargs="?",
        default="",
        help="Prefix or redirector to look for the files",
    )
    # parser.add_argument('--pretend'        , action='store_true', help = 'Read json files but, not execute the analysis')
    # parser.add_argument('--nworkers','-n' , default=8  , help = 'Number of workers')
    parser.add_argument(
        "--chunksize", "-s", default=100000, help="Number of events per chunk"
    )
    parser.add_argument(
        "--nchunks",
        "-c",
        default=None,
        help="You can choose to run only a number of chunks",
    )
    parser.add_argument(
        "--outname",
        "-o",
        default="histos",
        help="Name of the output file with histograms",
    )
    parser.add_argument(
        "--treename", default="Events", help="Name of the tree inside the files"
    )
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
        help="Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.",
    )
    parser.add_argument(
        "--processor",
        "-p",
        default="analysis_processor.py",
        help="Specify processor file name",
    )

    args = parser.parse_args()
    inputFile = args.inputFile
    executor = args.executor
    prefix = args.prefix
    # pretend     = args.pretend
    # nworkers    = int(args.nworkers)
    chunksize = int(args.chunksize)
    nchunks = int(args.nchunks) if args.nchunks is not None else args.nchunks
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

    # Check if we have valid options
    if executor not in known_executors:
        raise Exception(
            f'The "{executor}" executor is not known. Please specify an executor from the known executors ({known_executors}). Exiting.'
        )

    if executor in ["vr"]:
        # construct wq port range
        port = list(map(int, ports.split("-")))
        if len(port) < 1:
            raise ValueError("At least one port value should be specified.")
        if len(port) > 2:
            raise ValueError("More than one port range was specified.")
        if len(port) == 1:
            # convert single values into a range of one element
            port.append(port[0])

        # print(f"port: {port}")

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

        # print(f"\n\n redirector from yaml: {redirector} \n\n")

    flist = {}
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]["redirector"]
        flist[sname] = [(redirector + f) for f in samplesdict[sname]["files"]]

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

    # Run the processor and get the output
    tstart = time.time()

    if executor == "vr":

        mgr = vine.Manager(
            port=port,
            name=f"{os.environ['USER']}-vr-coffea",
        )
        mgr.tune("hungry-minimum", 1)
        mgr.enable_monitoring(watchdog=False)

        # Check if the X509 proxy file exists
        x509_proxy = f"/tmp/x509up_u{os.getuid()}"
        if not os.path.exists(x509_proxy):
            print(
                f"Warning: X509 proxy file {x509_proxy} does not exist. Setting to None."
            )
            x509_proxy = None
        else:
            shutil.copy(x509_proxy, "./proxy.pem")

        data = {}
        for sname in samplesdict.keys():
            data[sname] = preprocessing_for_vr(samplesdict[sname])

        print("\n\n samplesdict:")
        pprint.pprint(samplesdict)

        print("\nPreprocessing data with TaskVine...")
        preprocessed_data = preprocess(
            manager=mgr,
            data=data,
            tree_name="Events",
            timeout=30,
            max_retries=5,
            show_progress=True,
            batch_size=5,
            x509_proxy=x509_proxy,
            save_to_file=os.path.splitext(inputFile)[0],
        )

        # with open(f"{inputFile}_preprocessed.json", "w") as f:
        #     json.dump(preprocessed_data, f, indent=2)

        # print(f"\n\n preprocessed data saved to: {inputFile}_preprocessed.json \n\n")

        # print(f"\n\n proxy: {x509_proxy} \n\n")

        # VineReduce
        vr = CoffeaVineReduce(
            mgr,  # taskvine manager,
            data=preprocessed_data,
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
            # processors = processors,
            # accumulator=analysis_processor.AnalysisProcessor,
            # extra_files = [proc_file, "/users/hnelson2/ttbarEFT-coffea2025/ttbarEFT/params/channels.json", x509_proxy],
            extra_files=[
                proc_file,
                "/users/hnelson2/ttbarEFT-coffea2025/ttbarEFT/params/channels.json",
                "proxy.pem",
            ],
            schema=NanoAODSchema,
            max_task_retries=20,  # default=10
            step_size=600000,  # equivalent to chunksize, default=100k
            resources_processing={"cores": 1},
            resources_accumulating={"cores": 1},
            results_directory=results_dir,
            verbose=True,
            x509_proxy=x509_proxy,
        )
        vr.environment_variables["X509_USER_PROXY"] = "proxy.pem"

        hists = vr.compute()

        print("\n\n resulting hists: ")
        pprint.pprint(hists)

    elif executor == "iterative":

        print(f"samplesdict: {samplesdict} \n\n")

        flist = preprocessing_for_taskvine(samplesdict)
        proc_instance = analysis_processor.AnalysisProcessor(
            samples=samplesdict, lep_cat="em", wc_names_lst=wc_lst, hist_lst=hist_lst
        )
        exec_instance = processor.IterativeExecutor()
        runner = processor.Runner(
            exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks
        )
        hists = runner(
            fileset=flist, processor_instance=proc_instance, treename=treename
        )

    # Save Output ###
    outpath = "."
    out_pkl_file = os.path.join(outpath, f"{outname}.pkl.gz")
    print(f"\n\n Saving output to {out_pkl_file}")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(hists, fout)
        print("Done! \n\n")

    tend = time.time()
    print(f"\n\n Total processing time: {tend-tstart}")

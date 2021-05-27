#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import sleep
import os
import signal
from subprocess import Popen, PIPE, STDOUT
import re
import numpy as np
import pandas as pd


def read_image(row, reader):
    return reader.read(
        c=row["C_index"],
        t=row["T_index"],
        series=row["S_index"],
        z=row["Z_index"],
        rescale=False,
    )


def with_ipcluster(func):
    def wrapped(*args, **kwargs):
        #        print(args)
        #        print(kwargs.keys())
        if "ipcluster_nproc" in kwargs.keys():
            nproc = kwargs["ipcluster_nproc"]
        else:
            nproc = 1
        if "ipcluster_timeout" in kwargs.keys():
            timeout = kwargs["ipcluster_timeout"]
        else:
            timeout = 100
        command = ["ipcluster", "start", "--profile", "default", "--n", str(nproc)]
        try:
            print("starting ipcluster...")
            proc = Popen(command, stdout=PIPE, stderr=PIPE)
            i = 0
            while True:
                sleep(1)
                outs = proc.stderr.readline().decode("ascii")
                print(outs.replace("\n", ""))
                if "successfully" in outs:
                    break
                if i > timeout:
                    raise TimeoutError("ipcluster timeout")
                i = i + 1
            print("started.")
            res = func(*args, **kwargs)
        finally:
            print("terminating ipcluster...")
            #            os.kill(proc.pid, signal.SIGTERM)
            os.kill(proc.pid, signal.SIGINT)

    #            proc.terminate()
    #            proc.communicate()
    #        return res
    #        with Popen(command) as proc:
    #            print("starting ipcluster...")
    #            sleep(10)
    #            print("started.")
    #            try:
    #                res=func(*args,**kwargs)
    #            finally:
    #                os.kill(proc.pid, signal.SIGINT)
    ##                proc.terminate()
    #        return res
    return wrapped


def check_ipcluster_variable_defined(dview, name, timeout=10):
    for i in range(timeout):
        print(f"trying to find {name}...", i)
        try:
            dview.execute(f"print({name})")
            return
        except ipp.error.CompositeError:
            sleep(1)
            pass
    raise RuntimeError("check ipcluster timeout")


def send_variable(dview, name, variable, timeout=10):
    dview.push({name: variable})
    sleep(1)
    check_ipcluster_variable_defined(dview, name, timeout)


def parse_stitching_result(filename):
    pattern = r"rescaled_t([\d]+)_row([\d]+)_col([\d]+)_color([\d]+)\.tiff; ; \(([\-\d\.]+), ([\-\d\.]+)\)"
    names = ["t", "row", "col", "color", "pos_x", "pos_y"]
    parsed_result = []
    with open(filename, "r") as f:
        for line in f.readlines():
            res = re.search(pattern, line)
            if not res is None:
                parsed_result.append(dict(zip(names, res.groups())))
    stitching_result_df = pd.DataFrame.from_records(parsed_result)
    for k in names[:4]:
        stitching_result_df[k] = stitching_result_df[k].astype(np.int32)
    for k in names[4:]:
        stitching_result_df[k] = stitching_result_df[k].astype(np.float64)
    return stitching_result_df


def wrap_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

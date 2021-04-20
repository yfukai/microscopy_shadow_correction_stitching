#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import sleep
from subprocess import Popen

def read_image(row, reader):
    return reader.read(c=row["C_index"],
                       t=row["T_index"],
                       series=row["S_index"],
                       z=row["Z_index"],
                       rescale=False)


def with_ipcluster(func):
    def wrapped(*args,**kwargs):
        if "ipcluster_nproc" in kwargs.keys():
            nproc=kwargs["ipcluster_nproc"]
        else:
            nproc=1
        try:
            print("starting ipcluster...")
            proc=Popen(["ipcluster","start","--profile","default","--n",str(nproc)])
            sleep(10)
            print("started.")
            res=func(*args,**kwargs)
        finally:
            print("terminating ipcluster...")
            proc.terminate()
        return res
    return wrapped


def check_ipcluster_variable_defined(dview,name,timeout=10):
    for i in range(timeout):
        try:
            dview.execute(f"print({name})")
            return
        except ipp.error.CompositeError:
            sleep(1)
            pass
    raise RuntimeError("check ipcluster timeout")



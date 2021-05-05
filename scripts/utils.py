#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import sleep
import os
import signal
from subprocess import Popen

def read_image(row, reader):
    return reader.read(c=row["C_index"],
                       t=row["T_index"],
                       series=row["S_index"],
                       z=row["Z_index"],
                       rescale=False)


def with_ipcluster(func):
    def wrapped(*args,**kwargs):
#        print(args)
#        print(kwargs.keys())
        if "ipcluster_nproc" in kwargs.keys():
            nproc=kwargs["ipcluster_nproc"]
        else:
            nproc=1
        command=["ipcluster","start","--profile","default","--n",str(nproc)]
#        try:
#            print("starting ipcluster...")
#            proc=Popen(command)
#            sleep(10)
#            print("started.")
#            res=func(*args,**kwargs)
#        finally:
#            print("terminating ipcluster...")
##            os.kill(proc.pid, signal.SIGTERM)
#            proc.terminate()
#            proc.communicate()
#        return res
        with Popen(command) as proc:
            print("starting ipcluster...")
            sleep(10)
            print("started.")
            try:
                res=func(*args,**kwargs)
            finally:
                os.kill(proc.pid, signal.SIGINT)
#                proc.terminate()
        return res
    return wrapped


def check_ipcluster_variable_defined(dview,name,timeout=10):
    for i in range(timeout):
        print(f"trying to find {name}...",i)
        try:
            dview.execute(f"print({name})")
            return
        except ipp.error.CompositeError:
            sleep(1)
            pass
    raise RuntimeError("check ipcluster timeout")

def send_variable(dview,name,variable,timeout=10):
    dview.push({name:variable})
    sleep(1)
    check_ipcluster_variable_defined(dview,name,timeout)

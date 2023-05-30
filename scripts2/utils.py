import numpy as np
def percentile_normalize(image,qmin=0.1,qmax=99.9):
    q1,q2=np.percentile(image,[qmin,qmax])
    return np.clip((image-q1)/(q2-q1),0,1)
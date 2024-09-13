import skimage as ski
import numpy as np
import random

def get_phantom(dim):
    phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
    return ski.transform.resize(phantom, (dim, dim))

class random_weighted_norm:
    def __init__(self, img_size = 64, pos_var = 0.1, w_max = 2, r = None):
        x,y = np.meshgrid(*[np.linspace(-1,1, img_size) for _ in [0,1]])
        self.xy = np.stack([x,y])
        self.pos_var = pos_var
        self.w_max = w_max
        self.r = (0.3,0.6) if r is None else r
          
    def __call__(self,p=1):
        w = np.random.uniform(1., self.w_max, size=(2,))
        r = random.uniform(*self.r)
        m = np.random.normal(0, self.pos_var, size=(2,1,1))
        return self.weighted_norm(m, w, p, r)
    
    def weighted_norm(self, m, w, p, r):
        return 1. * (np.linalg.norm((self.xy - m) * w[:,None,None], axis=0, ord=p) < r)
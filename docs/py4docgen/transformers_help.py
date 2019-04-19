# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:41:29 2018

@author: Lionel Massoulard
"""

import numpy as np
import matplotlib.pylab as plt

y=np.exp(np.random.random(1000)*10);


from aikit.transformers import BoxCoxTargetTransformer

bc = BoxCoxTargetTransformer(None,ll = 0)

plt.subplot(211)
plt.plot(y)
plt.title("row data")

plt.subplot(212)
plt.plot(bc.target_transform(y))
plt.title("transformed data : log(1 + data)")


plt.cla()
ys = np.sort(y)
for ll in (0,0.1,0.2,0.3,0.4,0.5):
    bc = BoxCoxTargetTransformer(None,ll = ll)
    plt.plot(ys,bc.target_transform(ys),label = "ll = %2.2f" % ll)
plt.legend()
plt.xlabel("raw data")
plt.ylabel("transformed data")
plt.title("boxcox family of transformations")
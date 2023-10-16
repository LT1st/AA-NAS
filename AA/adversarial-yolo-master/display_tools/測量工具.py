# https://github.com/up42/image-similarity-measures

from image_similarity_measures.quality_metrics import rmse
import numpy as np

a = rmse(org_img=np.random.rand(3,2,1), pred_img=np.random.rand(3,2,1))

print(a)

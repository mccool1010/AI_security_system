import os
from deepface import DeepFace
import glob

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_images')
imgs = glob.glob(os.path.join(TEST_DIR, '*'))
if not imgs:
    print('NO_TEST_IMAGES')
    raise SystemExit(0)

img = imgs[0]
print('Using image:', img)
# Use enforce_detection=False to avoid exceptions on small or odd images
rep = DeepFace.represent(img_path=img, model_name='VGG-Face', enforce_detection=False)

# rep may be list/dict/ndarray depending on DeepFace version
import numpy as np
try:
    vec = np.array(rep).astype(float).flatten()
    print('embedding_len=', vec.shape)
except Exception:
    print('represent returned type:', type(rep))
    print('value:', rep)

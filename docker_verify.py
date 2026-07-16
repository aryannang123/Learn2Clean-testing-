"""Build-time verification script for Docker."""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import Learn2Clean_TFM as t
sys.modules.setdefault('learn2clean_v3', t)

from tabpfn import TabPFNClassifier
import numpy as np

clf = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
clf.fit(np.random.rand(40, 5), np.random.randint(0, 2, 40))
preds = clf.predict(np.random.rand(5, 5))
print('Build OK: TabPFN works, predictions:', preds)

import time
from inference import predict_mismatch, load_model
import glob

# warm up - this loads and caches the model once
load_model()
files = sorted(glob.glob("../data/demo_test_set/*.wav"))[:5]

times = []
for f in files:
    start = time.time()
    result = predict_mismatch(f)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"{f}: {elapsed:.2f}s")

print(f"\nAverage inference time: {sum(times)/len(times):.2f}s over {len(times)} files")

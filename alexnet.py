import argparse, time
from jetson_inference import imagenet
from jetson_utils import loadImage

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", help="input image(s)")
parser.add_argument("--topk", type=int, default=5)
parser.add_argument("--network", default="alexnet")
args = parser.parse_args()

net = imagenet(network=args.network)
print(f"Loaded {args.network}")

for path in args.images:
    img = loadImage(path)
    start = time.time()

    class_id, confidence = net.Classify(img)
    elapsed = (time.time() - start) * 1000

    print(f"\nImage: {path}")
    print(f" Top-1: {net.GetClassDesc(class_id)} ({confidence:.2f})")
    print(f" Time : {elapsed:.2f} ms")

    probs = net.GetLastOutputs()
    topk = probs.argsort()[-args.topk:][::-1]

    print(" Top-K:")
    for i in topk:
        print(f"  - {net.GetClassDesc(i)} ({probs[i]:.3f})")

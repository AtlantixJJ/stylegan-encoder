import os

gpus = [0, 1, 2, 3, 4, 5, 6, 7]
basecmd = "CUDA_VISIBLE_DEVICES=%d python encode_images.py ~/data reconstruction dlatent noise --start %d --end %d &"
for i in range(len(gpus)):
    os.system(basecmd % (gpus[i], i * 100, (i + 1) * 100))
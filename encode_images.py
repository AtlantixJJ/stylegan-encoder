import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import tensorflow as tf

#URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
PATH_FFHQ = 'cache/karras2019stylegan-ffhq-1024x1024.pkl'


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
parser.add_argument('src_dir', help='Directory with images for encoding')
parser.add_argument('generated_images_dir', help='Directory for storing generated images')
parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')
parser.add_argument('noise_dir', help='Directory for storing dlatent representations')

# for now it's unclear if larger batch leads to better performance/quality
parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
parser.add_argument('--start', type=int, default=0, help='Start from')
parser.add_argument('--end', type=int, default=100, help='Start from')

# Perceptual model params
parser.add_argument('--image_size', default=1024, help='Size of images for perceptual model', type=int)
parser.add_argument('--lr', default=0.01, help='Learning rate for perceptual model', type=float)
parser.add_argument('--iterations', default=500, help='Number of optimization steps for each batch', type=int)

# Generator params
parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
args, other_args = parser.parse_known_args()

ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
ref_images = list(filter(os.path.isfile, ref_images))
ref_images.sort()
ref_images = ref_images[args.start:args.end]

if len(ref_images) == 0:
    raise Exception('%s is empty' % args.src_dir)

os.makedirs(args.generated_images_dir, exist_ok=True)
os.makedirs(args.dlatent_dir, exist_ok=True)
os.makedirs(args.noise_dir, exist_ok=True)

# Initialize generator and perceptual model
tflib.init_tf()
#with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
with open(PATH_FFHQ, "rb") as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
    del generator_network, discriminator_network

generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)
perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)
perceptual_model.build_perceptual_model(generator.generated_image)
perceptual_model.setup(generator.dlatent_variable, generator.noise_variable, args.lr)
refs = [perceptual_model.ref_image] + perceptual_model.ref_img_features
min_op = perceptual_model.min_op
noise_min_op = perceptual_model.noise_min_op
loss = perceptual_model.loss
sess = tf.get_default_session()
sess.graph.finalize()

record = []
for images_batch in tqdm(split_to_batches(ref_images, args.batch_size),
    total=len(ref_images)//args.batch_size):
    names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

    # get reference feature
    refs_np = perceptual_model.get_reference_features(images_batch)

    best_loss = 0xffffffff
    losses = []
    generator.reset()
    pbar = tqdm(range(args.iterations), leave=False)
    for i in pbar:
        if i % 5 == 0:
            _, loss_np = sess.run([noise_min_op, loss], {t:v for t,v in zip(refs, refs_np)})
            sess.run([generator.clamp_noise_op])
        else:
            _, loss_np = sess.run([min_op, loss], {t:v for t,v in zip(refs, refs_np)})

        if loss_np < best_loss and i > args.iterations * 0.7:
            best_loss = loss_np
            v = generator.get_param()
            best_d = v[0]
            best_n = v[1:]
            best_image = generator.generate_images()
        losses.append(loss_np)
        pbar.set_description(' '.join(names)+' Loss: %.2f' % loss_np)
    print(' '.join(names), ' loss:', best_loss)

    # Generate images from found dlatents and save them
    img_name = names[0]
    img = PIL.Image.fromarray(best_image[0], 'RGB')
    img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
    np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), best_d)
    obj = np.zeros((len(best_n),), dtype="object")
    obj[:] = best_n
    np.save(os.path.join(args.noise_dir, f'{img_name}.npy'), obj)
    np.save(os.path.join(args.generated_images_dir, f'{img_name}.npy'), losses)

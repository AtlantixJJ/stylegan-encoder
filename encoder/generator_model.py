import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial
import os

def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size):
    return tf.get_variable('learnable_dlatents',
                           shape=(batch_size, 18, 512),
                           dtype='float32',
                           initializer=tf.initializers.random_normal())


class Generator:
    def __init__(self, model, batch_size, randomize_noise=False):
        self.batch_size = batch_size
        self.synthesis = model.components.synthesis
        self.mapping = model.components.mapping
        self.initial_dlatents_np = self.get_mean_dlatents()

        self.synthesis.run(
            self.initial_dlatents_np,
            randomize_noise=randomize_noise,
            minibatch_size=self.batch_size,
            custom_inputs=[
                partial(create_variable_for_generator,
                    batch_size=batch_size),
                partial(create_stub,
                batch_size=batch_size)],
            structure='fixed')

        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.dlatent_variable = next(v for v in tf.global_variables()
            if 'learnable_dlatents' in v.name)
        self.dlatents_input = tf.placeholder(tf.float32, shape=[None, 18, 512])
        self.noise_variable = [v for v in tf.global_variables() if 'noise' in v.name]
        print(self.noise_variable)
        self.noise_input = [tf.placeholder(tf.float32, shape=v.shape)
            for v in self.noise_variable]
        self.assign_dlatent_op = tf.assign(self.dlatent_variable, self.dlatents_input)
        self.assign_noise_op = [tf.assign(noise, noise_input)
            for noise, noise_input in zip(self.noise_variable, self.noise_input)]
        self.clamp_noise_op = [tf.assign(noise, tf.clip_by_value(noise, -1, 1))
            for noise in self.noise_variable]
        self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0')
        self.generated_image = tflib.convert_images_to_uint8(
            self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

    def get_mean_dlatents(self):
        if os.path.exists("face_average_w.npy"):
            self.avg_w = np.load("face_average_w.npy")
            return self.avg_w

        print("=> Calculating mean face w")
        self.rng = np.random.RandomState(1314)
        w_ = np.zeros((4096, 18, 512), dtype="float32")
        for i in range(4096):
            latent = self.rng.randn(1, 512)
            w_[i] = self.mapping.run(latent, None)[0]
        self.avg_w = w_.mean(0, keepdims=True)
        np.save("face_average_w.npy", self.avg_w)
        return self.avg_w

    def reset(self):
        dic = {self.dlatents_input : self.initial_dlatents_np}
        for noise in self.noise_input:
            dic[noise] = np.random.randn(*noise.shape)
        self.sess.run([self.assign_dlatent_op] + self.noise_input, dic)

    def get_param(self):
        return self.sess.run([self.dlatent_variable] + self.noise_variable)

    def generate_images(self, dlatents=None):
        if dlatents:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)

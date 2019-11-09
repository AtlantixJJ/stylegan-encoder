import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


class PerceptualModel:
    def __init__(self, img_size=1024, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layers = [1, 2, 8, 12]
        self.n_layers = len(self.layers)
        self.batch_size = batch_size

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

    def build_perceptual_model(self, image):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.input = vgg16.input
        self.outputs = [vgg16.layers[l].output for l in self.layers]
        self.perceptual_model = Model(self.input, self.outputs)
        image = preprocess_input(image)
        image_features = self.perceptual_model(image)
        N = image_features[0].shape[0]

        self.ref_image = tf.placeholder(tf.float32, shape=[N, self.img_size, self.img_size, 3])
        self.ref_img_features = [
            tf.placeholder(tf.float32, shape=image_features[i].shape)
                for i in range(self.n_layers)]

        losses = [tf.losses.mean_squared_error(ref, img)
            for ref,img in zip(self.ref_img_features, image_features)]
        self.loss = sum(losses)
        self.loss += tf.losses.mean_squared_error(self.ref_image, image)

    def get_reference_features(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        #image_features = self.perceptual_model.predict_on_batch(loaded_image)
        image_features = self.sess.run(self.outputs, {self.input: loaded_image})
        return [load_images] + image_features

    def setup(self, vars_to_optimize, learning_rate):
        self.vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.min_op = self.optimizer.minimize(self.loss, var_list=[self.vars_to_optimize])
        self.sess.run(tf.variables_initializer(self.optimizer.variables()))

    def optimize(self, iterations=500):
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        for _ in range(iterations):
            _, loss = self.sess.run([self.min_op, self.loss], options=run_options)
            yield loss

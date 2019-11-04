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
        outputs = [vgg16.layers[l].output for l in self.layers]
        self.perceptual_model = Model(vgg16.input, outputs)
        image = preprocess_input(image)
        image_features = self.perceptual_model(image)

        self.ref_image = tf.get_variable('ref_img',
            shape=[image_features[0].shape[0], self.img_size, self.img_size, 3],
            dtype='float32',
            initializer=tf.initializers.zeros())
        self.ref_img_features = [
            tf.get_variable('ref_img_features_%d' % i,     
                shape=image_features[i].shape,
                dtype='float32',
                initializer=tf.initializers.zeros()) for i in range(self.n_layers)]
        self.sess.run([f.initializer for f in self.ref_img_features])

        losses = [tf.losses.huber_loss(ref, img)
            for ref,img in zip(self.ref_img_features, image_features)]
        self.loss = sum(losses) / len(losses)
        self.loss += tf.losses.huber_loss(self.ref_image, image)

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = self.perceptual_model.predict_on_batch(loaded_image)
        self.sess.run(tf.assign(self.ref_image, loaded_image))
        self.sess.run([tf.assign(ref, img)
            for ref,img in zip(self.ref_img_features, image_features)])

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1.):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run([tf.global_variable_initializer()])
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield loss


import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp
import Models
import datetime
import time
import numpy as np


def train(encoder: Models.Encoder, decoder: Models.Decoder, svm: Models.Svm, dataset):
    @tf.function
    def _gan_train_step(encoder: kr.Model, encoder_optimizer: kr.optimizers.Optimizer,
                        decoder: kr.Model, decoder_optimizer: kr.optimizers.Optimizer,
                        latent_var_trace: tf.Variable, data, **kwargs):
        real_images = data['image']
        batch_size = real_images.shape[0]
        latent_vectors = hp.latent_dist_func(batch_size)

        if hp.is_dls:
            latent_scale_vectors = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32')
                                          * decoder.latent_var_trace / tf.reduce_sum(decoder.latent_var_trace))[tf.newaxis]
        else:
            latent_scale_vectors = tf.ones([1, hp.latent_vector_dim])
        fake_images = decoder(latent_vectors * latent_scale_vectors)

        with tf.GradientTape() as dis_tape:
            with tf.GradientTape() as reg_tape:
                reg_tape.watch(real_images)
                real_adv_values, _, _ = encoder(real_images)
            reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(real_adv_values, real_images)), axis=[1, 2, 3])
            fake_adv_values, rec_latent_vectors, rec_latent_logvars = encoder(fake_images)

            latent_diff = tf.square((latent_vectors - rec_latent_vectors) * latent_scale_vectors)

            if hp.use_logvar:
                enc_losses = tf.reduce_mean(rec_latent_logvars + latent_diff / (tf.exp(rec_latent_logvars) + 1e-7), axis=-1)
            else:
                enc_losses = tf.reduce_mean(latent_diff, axis=-1)

            dis_adv_losses = tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values)
            dis_loss = tf.reduce_mean(dis_adv_losses + hp.enc_weight * enc_losses + hp.reg_weight * reg_losses)

        encoder_optimizer.minimize(dis_loss, encoder.trainable_variables, tape=dis_tape)
        rec_latent_traces = rec_latent_vectors

        with tf.GradientTape() as gen_tape:
            latent_vectors = hp.latent_dist_func(batch_size)
            fake_images = decoder(latent_vectors * latent_scale_vectors)

            fake_adv_values, rec_latent_vectors, rec_latent_logvars = encoder(fake_images)
            latent_diff = tf.square((latent_vectors - rec_latent_vectors) * latent_scale_vectors)

            if hp.use_logvar:
                enc_losses = tf.reduce_mean(rec_latent_logvars + latent_diff / (tf.exp(rec_latent_logvars) + 1e-7), axis=-1)
            else:
                enc_losses = tf.reduce_mean(latent_diff, axis=-1)
            gen_adv_losses = tf.nn.softplus(-fake_adv_values)

            gen_loss = tf.reduce_mean(gen_adv_losses + hp.enc_weight * enc_losses)

        decoder_optimizer.minimize(gen_loss, decoder.trainable_variables, tape=gen_tape)

        rec_latent_traces = tf.concat([rec_latent_traces, rec_latent_vectors], axis=0)
        latent_var_trace.assign(latent_var_trace * hp.latent_var_decay_rate +
                                tf.reduce_mean(tf.square(rec_latent_traces), axis=0) * (1.0 - hp.latent_var_decay_rate))

        results = {'real_adv_values': real_adv_values, 'fake_adv_values': fake_adv_values,
                   'enc_losses': enc_losses, 'reg_losses': reg_losses}

        return results

    @tf.function
    def _autoencoder_train_step(encoder: kr.Model, encoder_optimizer: kr.optimizers.Optimizer,
                                decoder: kr.Model, decoder_optimizer: kr.optimizers.Optimizer,
                                data, **kwargs):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            rec_images = decoder(encoder(real_images)[1])

            rec_losses = tf.reduce_mean(tf.square(rec_images - real_images), axis=[1, 2, 3])
            rec_loss = tf.reduce_mean(rec_losses)
        encoder_optimizer.minimize(rec_loss, encoder.trainable_variables, tape=tape)
        decoder_optimizer.minimize(rec_loss, decoder.trainable_variables, tape=tape)

        results = {'rec_losses': rec_losses}

        return results

    @tf.function
    def _classifier_train_step(encoder: kr.Model, encoder_optimizer: kr.optimizers.Optimizer,
                               svm: kr.Model, svm_optimizer: kr.optimizers.Optimizer,
                               data, **kwargs):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            real_labels = data['label']
            predict_logits = svm(tf.nn.relu(encoder(real_images)[1]))
            ce_losses = tf.losses.categorical_crossentropy(real_labels, predict_logits, from_logits=True)
            ce_loss = tf.reduce_mean(ce_losses)

        encoder_optimizer.minimize(ce_loss, encoder.trainable_variables, tape=tape)
        svm_optimizer.minimize(ce_loss, svm.trainable_variables, tape=tape)

        results = {'ce_losses': ce_losses}
        return results

    print('\ntraining...')
    total_start = time.time()
    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        epoch_start = time.time()

        results = {}

        for data in dataset:
            args = {'encoder': encoder.model, 'encoder_optimizer': encoder.optimizer,
                    'decoder': decoder.model, 'decoder_optimizer': decoder.optimizer,
                    'svm': svm.model, 'svm_optimizer': svm.optimizer,
                    'latent_var_trace': decoder.latent_var_trace, 'data': data}
            if hp.train_gan:
                batch_results = _gan_train_step(**args)
            elif hp.train_autoencoder:
                batch_results = _autoencoder_train_step(**args)
            elif hp.train_classifier:
                batch_results = _classifier_train_step(**args)
            else:
                raise AssertionError

            for key in batch_results:
                try:
                    results[key].append(batch_results[key])
                except KeyError:
                    results[key] = [batch_results[key]]

        encoder.optimizer.lr = encoder.optimizer.lr * hp.lr_decay_rate
        decoder.optimizer.lr = decoder.optimizer.lr * hp.lr_decay_rate
        svm.optimizer.lr = svm.optimizer.lr * hp.lr_decay_rate

        temp_results = {}
        for key in results:
            mean, variance = tf.nn.moments(tf.concat(results[key], axis=0), axes=0)
            temp_results[key + '_mean'] = mean
            temp_results[key + '_variance'] = variance
        results = temp_results

        for key in results:
            print('%-30s:' % key, '%13.6f' % np.array(results[key]))
        print('epoch time: ', time.time() - epoch_start, '\n')
    print('total time: ', time.time() - total_start, '\n')
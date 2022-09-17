import tensorflow.keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as hp
from scipy.linalg import sqrtm
import numpy as np
import scipy.stats as ss
from sklearn.metrics import auc
import Models
import time
import datetime

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


@tf.function
def _get_feature_samples(decoder: kr.Model, latent_scale_vector, real_images):
    batch_size = real_images.shape[0]
    latent_vectors = hp.latent_dist_func(batch_size)

    fake_images = tf.clip_by_value(
        decoder(latent_vectors * latent_scale_vector[tf.newaxis]),
        clip_value_min=-1, clip_value_max=1)
    real_images = tf.tile(real_images, [1, 1, 1, 3])
    fake_images = tf.tile(fake_images, [1, 1, 1, 3])
    real_images = tf.image.resize(real_images, [299, 299])
    fake_images = tf.image.resize(fake_images, [299, 299])

    real_features = inception_model(real_images)
    fake_features = inception_model(fake_images)

    return real_features, fake_features


def _get_features(decoder: kr.Model, latent_scale_vector, test_dataset: tf.data.Dataset):
    real_features = []
    fake_features = []

    for data in test_dataset:
        real_features_batch, fake_features_batch = _get_feature_samples(decoder, latent_scale_vector, data['image'])
        real_features.append(real_features_batch)
        fake_features.append(fake_features_batch)

    real_features = tf.concat(real_features, axis=0)
    fake_features = tf.concat(fake_features, axis=0)

    return real_features, fake_features


def get_fid(decoder: kr.Model, latent_scale_vector, test_dataset: tf.data.Dataset):
    real_features, fake_features = _get_features(decoder, latent_scale_vector, test_dataset)
    real_features_mean = tf.reduce_mean(real_features, axis=0)
    fake_features_mean = tf.reduce_mean(fake_features, axis=0)

    mean_difference = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_features), tfp.stats.covariance(fake_features)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_difference = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_difference + cov_difference

    return fid


def evaluate_fake_rec(encoder: kr.Model, decoder: kr.Model, latent_scale_vector, id_dataset: tf.data.Dataset):
    average_psnr = []
    average_ssim = []
    for _ in id_dataset:
        latent_vectors = hp.latent_dist_func(hp.batch_size)
        fake_images = tf.clip_by_value(
            decoder(latent_vectors * latent_scale_vector[tf.newaxis]),
            clip_value_min=-1, clip_value_max=1)

        _, rec_latent_vectors = encoder(fake_images)
        rec_images = tf.clip_by_value(
            decoder(rec_latent_vectors * latent_scale_vector[tf.newaxis]),
            clip_value_min=-1, clip_value_max=1)

        average_psnr.append(tf.reduce_mean(tf.image.psnr(fake_images, rec_images, max_val=2.0)))
        average_ssim.append(tf.reduce_mean(tf.image.ssim(fake_images, rec_images, max_val=2.0)))

    return tf.reduce_mean(average_psnr), tf.reduce_mean(average_ssim)


def evaluate_real_rec(encoder: kr.Model, decoder: kr.Model, latent_scale_vector, id_dataset: tf.data.Dataset):
    average_psnr = []
    average_ssim = []
    for data in id_dataset:
        real_images = data['image']
        _, rec_latent_vectors = encoder(real_images)
        rec_images = tf.clip_by_value(
            decoder(rec_latent_vectors * latent_scale_vector[tf.newaxis]),
            clip_value_min=-1, clip_value_max=1)

        average_psnr.append(tf.reduce_mean(tf.image.psnr(real_images, rec_images, max_val=2.0)))
        average_ssim.append(tf.reduce_mean(tf.image.ssim(real_images, rec_images, max_val=2.0)))

    return tf.reduce_mean(average_psnr), tf.reduce_mean(average_ssim)


def get_accuracy(encoder: kr.Model, svm: kr.Model, id_dataset: tf.data.Dataset):
    total_count = 0
    wrong_count = 0
    for data in id_dataset:
        real_labels = tf.argmax(data['label'], axis=-1)
        predict_labels = tf.argmax(svm(tf.nn.relu(encoder(data['image'])[1])), axis=-1)

        batch_size = real_labels.shape[0]
        wrong = tf.math.count_nonzero(real_labels - predict_labels)

        total_count += batch_size
        wrong_count += wrong

    return (total_count - wrong_count) / total_count


def get_nll_ood_scores(encoder: kr.Model, latent_var_trace, test_dataset: tf.data.Dataset):
    scores = []
    for data in test_dataset:
        real_images = data['image']
        batch_size = real_images.shape[0]
        _, rec_latent_vectors = encoder(real_images)

        for i in range(batch_size):
            log_probs = -tf.math.log(tf.cast(ss.norm.pdf(rec_latent_vectors[i], scale=tf.sqrt(latent_var_trace)), 'float32'))
            scores.append(tf.reduce_sum(log_probs))

    return tf.convert_to_tensor(scores)


def get_energy_ood_scores(feature_vectors_set, svm: kr.Model, temperature, activation_threshold):
    scores_sets = []
    for feature_vectors in feature_vectors_set:
        logits = svm(tf.math.minimum(feature_vectors, activation_threshold))
        scores = -temperature * tf.math.log(tf.reduce_sum(tf.exp(logits / temperature), axis=-1))
        scores_sets.append(scores)

    return tf.concat(scores_sets, axis=0)


def get_rec_ood_scores(encoder: kr.Model, decoder: kr.Model, latent_scale_vector, dataset: tf.data.Dataset):
    scores_sets = []
    for data in dataset:
        real_images = data['image']
        rec_images = decoder(encoder(real_images)[1] * latent_scale_vector[tf.newaxis])
        scores = tf.reduce_mean(tf.square(real_images - rec_images), axis=[1, 2, 3])
        scores_sets.append(scores)

    return tf.concat(scores_sets, axis=0)


def get_auroc(id_ood_scores, ood_ood_scores):
    fprs = []
    tprs = []
    ood_thresholds = tf.sort(id_ood_scores, direction='DESCENDING')[::10]
    for ood_threshold in ood_thresholds:
        fprs.append(tf.cast(tf.math.count_nonzero(tf.where(id_ood_scores > ood_threshold, 1.0, 0.0)), 'float32') / id_ood_scores.shape[0])
        tprs.append(tf.cast(tf.math.count_nonzero(tf.where(ood_ood_scores > ood_threshold, 1.0, 0.0)), 'float32') / ood_ood_scores.shape[0])
    fprs.append(tf.constant(1.0))
    tprs.append(tf.constant(1.0))
    return auc(fprs, tprs)


def evaluate_gan(encoder: Models.Encoder, decoder: Models.Decoder, id_dataset, ood_datasets):
    results = {}
    if hp.is_dls:
        latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32')
                                      * decoder.latent_var_trace / tf.reduce_sum(decoder.latent_var_trace))
    else:
        latent_scale_vector = tf.ones([hp.latent_vector_dim])
    fake_psnr, fake_ssim = evaluate_fake_rec(encoder.model, decoder.model, latent_scale_vector, id_dataset)
    results['fake_psnr'] = fake_psnr
    results['fake_ssim'] = fake_ssim

    real_psnr, real_ssim = evaluate_real_rec(encoder.model, decoder.model, latent_scale_vector, id_dataset)
    results['real_psnr'] = real_psnr
    results['real_ssim'] = real_ssim

    fid = get_fid(decoder.model, latent_scale_vector, id_dataset)
    results['fid'] = fid

    nll_id_ood_scores = get_nll_ood_scores(encoder.model, decoder.latent_var_trace, id_dataset)
    rec_id_ood_scores = get_rec_ood_scores(encoder.model, decoder.model, latent_scale_vector, id_dataset)
    for key in ood_datasets:
        nll_ood_ood_scores = get_nll_ood_scores(encoder.model, decoder.latent_var_trace, ood_datasets[key])
        results[key + '_nll_auroc'] = get_auroc(nll_id_ood_scores, nll_ood_ood_scores)
        rec_ood_ood_scores = get_rec_ood_scores(encoder.model, decoder.model, latent_scale_vector, ood_datasets[key])
        results[key + '_rec_auroc'] = get_auroc(rec_id_ood_scores, rec_ood_ood_scores)

    return results


def evaluate_autoencoder(encoder: Models.Encoder, decoder: Models.Decoder, id_dataset, ood_datasets):
    results = {}

    real_psnr, real_ssim = evaluate_real_rec(encoder.model, decoder.model, tf.ones([hp.latent_vector_dim]), id_dataset)
    results['real_psnr'] = real_psnr
    results['real_ssim'] = real_ssim

    rec_id_ood_scores = get_rec_ood_scores(encoder.model, decoder.model, tf.ones([hp.latent_vector_dim]), id_dataset)
    for key in ood_datasets:
        rec_ood_ood_scores = get_rec_ood_scores(encoder.model, decoder.model, tf.ones([hp.latent_vector_dim]), ood_datasets[key])
        results[key + '_rec_auroc'] = get_auroc(rec_id_ood_scores, rec_ood_ood_scores)

    return results


def evaluate_classifier(encoder: Models.Encoder, svm: Models.Svm, id_dataset, ood_datasets):
    results = {}

    accuracy = get_accuracy(encoder.model, svm.model, id_dataset)
    results['accuracy'] = accuracy

    id_feature_vectors_set = [tf.nn.relu(encoder.model(data['image'])[1]) for data in id_dataset]
    sorted_id_feature_values = tf.sort(tf.reshape(tf.concat(id_feature_vectors_set, axis=0), [-1]))

    for key in ood_datasets:
        ood_feature_vectors_set = [tf.nn.relu(encoder.model(data['image'])[1]) for data in ood_datasets[key]]

        for react_p in hp.react_ps:
            activation_threshold = sorted_id_feature_values[round((sorted_id_feature_values.shape[0] - 1) * react_p)]

            for temperature in hp.temperatures:
                energy_id_ood_scores = get_energy_ood_scores(id_feature_vectors_set, svm.model, temperature, activation_threshold)
                energy_ood_ood_scores = get_energy_ood_scores(ood_feature_vectors_set, svm.model, temperature, activation_threshold)

                results[key + '_energy_t_' + str(temperature) + '_p_' + str(react_p) + '_auroc'] = get_auroc(energy_id_ood_scores, energy_ood_ood_scores)

    return results


def evaluate(encoder: Models.Encoder, decoder: Models.Decoder, svm: Models.Svm, id_dataset, ood_datasets):
    print(datetime.datetime.now())
    print('\nevaluating...')
    start = time.time()
    if hp.train_gan:
        results = evaluate_gan(encoder, decoder, id_dataset, ood_datasets)
    elif hp.train_autoencoder:
        results = evaluate_autoencoder(encoder, decoder, id_dataset, ood_datasets)
    elif hp.train_classifier:
        results = evaluate_classifier(encoder, svm, id_dataset, ood_datasets)
    else:
        raise AssertionError
    for key in results:
        print('%-50s:' % key, '%13.6f' % results[key])
    print('evaluate time: ', time.time() - start, '\n')
    return results
import os
import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import tensorflow_datasets as tfds
import numpy as np


def _to_array(dataset):
    images = []
    labels = []
    for data in dataset['train']:
        images.append(data['image'])
        labels.append(data['label'])
    for data in dataset['test']:
        images.append(data['image'])
        labels.append(data['label'])

    images = tf.pad(tf.convert_to_tensor(images), [[0, 0], [2, 2], [2, 2], [0, 0]])
    images = tf.cast(images, dtype='float32') / 127.5 - 1.0
    labels = tf.one_hot(tf.convert_to_tensor(labels), depth=hp.class_size)

    return images, labels


def load_id_dataset(i_start, i_end):
    dataset = tfds.load(hp.id_dataset)
    images, labels = _to_array(dataset)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        {'image': tf.concat([images[:i_start], images[i_end:]], axis=0),
         'label': tf.concat([labels[:i_start], labels[i_end:]], axis=0)}
    ).batch(hp.batch_size).shuffle(1000).prefetch(tf.data.AUTOTUNE)

    id_dataset = tf.data.Dataset.from_tensor_slices(
        {'image': images[i_start: i_end], 'label': labels[i_start: i_end]}
    ).batch(hp.batch_size).prefetch(tf.data.AUTOTUNE)

    if not os.path.exists('samples'):
        os.makedirs('samples')
    kr.preprocessing.image.save_img(
        path='samples/id_samples.png',
        x=np.hstack([np.vstack(images[hp.save_image_size * i:hp.save_image_size * (i + 1)])
                     for i in range(hp.save_image_size)]))

    return train_dataset, id_dataset


def load_ood_datasets(i_start, i_end):
    id_images, _ = _to_array(tfds.load(hp.id_dataset))
    ood_datasets = {}
    ood_samples = []

    for dataset_name in hp.ood_datasets:
        ood_images, _ = _to_array(tfds.load(dataset_name))
        for ood_intensity in hp.ood_intensities:
            interpolate_ood_images = ood_images * ood_intensity + id_images * (1.0 - ood_intensity)
            ood_datasets[dataset_name + '_k_' + str(ood_intensity)] = tf.data.Dataset.from_tensor_slices({
                'image': interpolate_ood_images[i_start: i_end]}).batch(hp.batch_size).prefetch(tf.data.AUTOTUNE)
            ood_samples.append(np.vstack(interpolate_ood_images[:hp.save_image_size]))

    if not os.path.exists('samples'):
        os.makedirs('samples')
    kr.preprocessing.image.save_img(
        path='samples/ood_samples.png',
        x=np.hstack(ood_samples))

    return ood_datasets

import numpy as np
import HyperParameters as hp
import Train
import Models
import Dataset
import Evaluate


def main():
    total_results = {}
    for i in range(hp.fold_size):
        d_start = int(70000 / hp.fold_size * i)
        d_end = int(70000 / hp.fold_size * (i + 1))

        train_dataset, id_dataset = Dataset.load_id_dataset(d_start, d_end)
        ood_datasets = Dataset.load_ood_datasets(d_start, d_end)

        encoder = Models.Encoder()
        decoder = Models.Decoder()
        svm = Models.Svm()

        Train.train(encoder, decoder, svm, train_dataset)

        encoder.save()
        decoder.save()
        svm.save()

        results = Evaluate.evaluate(encoder, decoder, svm, id_dataset, ood_datasets)

        for key in results:
            try:
                total_results[key].append(results[key])
            except KeyError:
                total_results[key] = [results[key]]

    if hp.train_gan:
        if hp.is_dls:
            file_name = 'DLSGAN.txt'
        else:
            file_name = 'InfoGAN.txt'
    elif hp.train_autoencoder:
        file_name = 'Autoencoder.txt'
    elif hp.train_classifier:
        file_name = 'Classifier.txt'
    else:
        raise AssertionError

    with open(file_name, 'w') as file:
        for key in total_results:
            average_result = np.mean(np.array(total_results[key]))
            print(key, ':', average_result)
            file.write(key + ':' + str(average_result) + '\n')


main()

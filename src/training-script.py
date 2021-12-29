import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
from azureml.core import Run
import argparse

# definice argumentů jež předává control-script.py
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01)
parser.add_argument('--epochs', type=int, dest='epochs', default=1)
parser.add_argument('--data-path', type=str, dest='data_path')
parser.add_argument('--run-num', type=int, dest='run_num')

args = parser.parse_args() # získá argumenty


run = Run.get_context()
run.tag("run number", args.run_num)


# definice modelu
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


# kompilace modelu
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# načtení datasetu
(train_images, train_labels), (test_images, test_labels) = tfds.load(name="cifar10", split=["train", "test"], data_dir=args.data_path, download=True, as_supervised=True, batch_size=-1)


# callback logující všechny metriky modelu po konci každé epochy. další možnosti: https://www.tensorflow.org/guide/keras/custom_callback
class AzureLogMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        metrics = list(logs.keys())
        for metric in metrics:
            run.log(metric, logs[metric]) 
        
        print("End epoch {} of training; got log keys: {}".format(epoch, metrics))


# spuštění trénování
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=args.epochs,
                    callbacks=[AzureLogMetrics()]
                    )

model.save_weights("./outputs/trained_model.h5")
print("YO THE MODEL HAS BEEN SAVED")




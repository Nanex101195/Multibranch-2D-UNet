import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

class Generator(tf.keras.utils.Sequence):
    def __init__(self, input_gen1, input_gen2, input_gen3, target_gen):
        self.gen1 = input_gen1
        self.gen2 = input_gen2
        self.gen3 = input_gen3
        self.gen4 = target_gen

        assert len(input_gen1) == len(input_gen2) == len(input_gen3) == len(target_gen)

    def __len__(self):
        return len(self.gen1)

    def __getitem__(self, i):
        x1 = self.gen1[i]
        x2 = self.gen2[i]
        x3 = self.gen3[i]
        y = self.gen4[i]
  

        return [x1, x2, x3], y

    def on_epoch_end(self):
        self.gen1.on_epoch_end()
        self.gen2.on_epoch_end()
        self.gen3.on_epoch_end()
        self.gen4.on_epoch_end()
        self.gen2.index_array = self.gen1.index_array
        self.gen3.index_array = self.gen1.index_array
        self.gen4.index_array = self.gen1.index_array


def datagen_train():
    data_gen_args_train = dict(rescale=1./255,  #needed to get values between 0 and 1 for pixel intensities
                               rotation_range=360,
                               horizontal_flip=True,
                               vertical_flip=True,
                               width_shift_range=0.3,
                               height_shift_range=0.3,
                               brightness_range=[0.5, 1.5],
                               zoom_range=[0.5, 1.5])
    data_gen_train = ImageDataGenerator(**data_gen_args_train)
    return data_gen_train


def datagen_val():
    data_gen_args_val = dict(rescale=1. / 255)
    data_gen_val = ImageDataGenerator(**data_gen_args_val)
    return data_gen_val


def datagenerator(gen, folder, IMG_SIZE, BATCH_SIZE, SEED):
    datagen_flow = gen.flow_from_directory(folder, target_size=IMG_SIZE, shuffle=True, class_mode=None,
                                           color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return datagen_flow
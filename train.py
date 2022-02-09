import warnings
warnings.filterwarnings('ignore')

from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.compat.v1 import InteractiveSession, ConfigProto
from tensorflow.compat.v1.keras.backend import set_session # 수정
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model # 수정
from keras.models import Model, load_model
from __future__ import print_function
from data_manager import DataManager
from generator import Generator
from evaluator import Evaluator
from config import configs
from models import NIC

import tensorflow.compat.v1 as tf # 수정
import matplotlib.pyplot as plt
import keras, h5py, pickle
from tqdm import tqdm
import os, tensorflow # 수정
import pandas as pd
import numpy as np

tensorflow.random.set_seed(2019)
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



# 하이퍼 파라미터 설정
num_epochs = 100
batch_size = 128

# 학습 이미지 경로
image_path='./dataset/'

# 이미지 추출 CNN layer 선택
cnn_extractor='resnet50'

# 추출된 이미지 feature 저장 파일 이름
object_image_features_filename=f'{cnn_extractor}_image_name_to_features.h5'

# 이미지 meta data 저장 경로
root_path = './dataset/'
captions_filename = root_path + 'image_text_tag.csv'

# 학습 이미지 feature 추출, 학습에 사용할 해시태그 형태로 가공
data_manager = DataManager(data_filename=captions_filename,
                            max_caption_length=30,
                            word_frequency_threshold=2,
                            extract_image_features=True, # 처음 시작에만 True
                            cnn_extractor=cnn_extractor,
                            image_directory=image_path,
                            split_data=True,
                            dump_path=root_path + 'preprocessed_data/',
                           sep=',')

data_manager.preprocess()

print(f">>> 해시태그 별 빈도수 : {data_manager.word_frequencies[0:20]}")

# 텍스트 토큰화 및 모델 학습 generator 생성
preprocessed_data_path = root_path + 'preprocessed_data/'
generator = Generator(data_path=preprocessed_data_path,
                      batch_size=batch_size,image_features_filename=object_image_features_filename)

num_training_samples =  generator.training_dataset.shape[0]
num_validation_samples = generator.validation_dataset.shape[0]
print(f'>>> 학습 샘플 수 : {num_training_samples}')
print(f'>>> 평가 샘플 수 : {num_validation_samples}')

print(f">>> 학습에 사용한 해시태그 수 : {generator.VOCABULARY_SIZE}")
print(f">>> 이미지 feature shape : {generator.IMG_FEATS}")

# 모델 생성
model =  NIC(max_token_length=generator.MAX_TOKEN_LENGTH,
            vocabulary_size=generator.VOCABULARY_SIZE,
            tweet_max_len=configs['tweet_max_len'],
            tweet_max_words=configs['tweet_max_words'],
            rnn='gru',
            num_image_features=generator.IMG_FEATS,
            hidden_size=256,
            embedding_size=128
            )

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

print(model.summary())
print(f'>>> Number of parameters : {model.count_params()}')
plot_model(model,show_shapes=True,to_file='NIH.png')

training_history_filename = preprocessed_data_path + 'training_hashtag_history.log'
csv_logger = CSVLogger(training_history_filename, append=False)
model_names = ('trained_models/image_text/' +
               'hashtag_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)

callbacks = [csv_logger, model_checkpoint, reduce_learning_rate]

history=model.fit(x=generator.flow(mode='train'), # 추가 (model.fit 도 generator 지원)
                    steps_per_epoch=num_training_samples // batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='validation'),
                    validation_steps=num_validation_samples // batch_size)

# 모델 저장
model.save("./trained_models/image_text/weights_final.hdf5")

# list all data in history
print(history.history.keys())

# Accuracy
plt.figure(figsize=(20,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./results/acc_val_acc.png',dpi=300)
plt.cla()
plt.clf()
plt.close()

# Loss
plt.figure(figsize=(20,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./results/loss_val_loss.png',dpi=300)
plt.show()

# Evaluator - precision, recall 측정용 파일 생성
evaluator = Evaluator(model, data_path=preprocessed_data_path,images_path=image_path,image_name_to_features_filename=object_image_features_filename)
evaluator.write_caption_after_train()
evaluator.display_caption_after_train()
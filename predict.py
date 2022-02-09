import warnings
warnings.filterwarnings('ignore')

from tensorflow.compat.v1 import InteractiveSession, ConfigProto
from tensorflow.compat.v1.keras.backend import set_session
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model, load_model
from data_manager import DataManager
from models import NIC

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import keras, h5py, pickle
from tqdm import tqdm
import os, tensorflow
import pandas as pd
import numpy as np

tensorflow.random.set_seed(2019)
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


class Predict_AMNN(object):
    def __init__(self,
            train_data_path = './dataset/preprocessed_data/',
            extract_img_feature = False,
            target_img_path = './test_dataset/',
            cnn_extractor = 'resnet50',
            target_data_path='./test_dataset/preprocessed_data/',
            target_csv_path = "./test_dataset/image_text_tag.csv",
            word_to_id_filename='word_to_id.p',
            id_to_word_filename='id_to_word.p',
            image_name_to_features_filename='resnet50_image_name_to_features.h5',
            model_file_path = "./trained_models/image_text/weights_final.hdf5"
            ):
      
        ### 0. 이미지 피처 추출을 안한 경우 피처 추출 ###
        
        if (extract_img_feature):
            self.target_data_path = target_data_path
            self.target_img_path = target_img_path
            self.cnn_extractor = cnn_extractor
            self.img_feature()


        ### 1. 학습 전 모델 관련 정보 설정 ###
        
        # 모델 학습에 사용된 해시태그 정보
        self.word_to_id = pickle.load(open(train_data_path + word_to_id_filename, 'rb'))
        self.id_to_word = pickle.load(open(train_data_path + id_to_word_filename, 'rb'))
        self.VOCABULARY_SIZE = len(self.word_to_id)
        self.tweet_max_words = 50000
        self.tweet_max_len = 300
        
        # 학습 모델 호출
        self.load_custom_model(train_data_path, model_file_path) 


        ### 2. 예측 데이터셋 준비 ###

        # 예측에 사용할 csv  
        self.target_csv = pd.read_csv(target_csv_path).astype(str)

        # 예측에 사용할 parameter 정보
        self.target_data_path = target_data_path
        data_logs = self._load_log_file()
        self.BOS = str(data_logs['BOS:'])
        self.EOS = str(data_logs['EOS:'])
        self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
        
        # 타겟 이미지 피처 로드
        self.image_names_to_features = h5py.File(target_data_path + image_name_to_features_filename)
        
        # 사용자 입력 텍스트(tweets) 토큰화
        self.tweets_to_token()


    def img_feature(self):
        self.data_manager = DataManager(data_filename=target_csv_path,
                            cnn_extractor = self.cnn_extractor,
                            image_directory = self.target_img_path,
                            dump_path = self.target_data_path,
                            )
        
        self.data_manager.original_directory = os.getcwd()
        data = pd.read_csv(target_csv_path)
        
        if (len(data.columns) < 3):
            data['caption'] = None # 모델 학습을 위해 이렇게 코드가 작성 되어 있음, 이미지 피처 추출과는 무관
        data = data.values
        self.data_manager.image_files = data[:, 0]
        self.data_manager.captions = data[:, 2]
        self.data_manager.tweets=data[:,1]

        self.data_manager.IMG_FEATS = 2048 # resnet out shape

        self.data_manager.get_image_features(self.target_img_path)

        self.data_manager.move_to_path()

        image_features_to_h5_name = f"{self.cnn_extractor}_image_name_to_features.h5"
        if os.path.isfile(image_features_to_h5_name):
            os.remove(image_features_to_h5_name)

        self.data_manager.write_image_features_to_h5()
        print(f"{self.cnn_extractor}_image_name_to_features.h5 파일 생성 완료")
        self.data_manager.move_path_back()


    # 해시태그 사용 빈도수 확인 
    def construct_dictionaries(self):
        self.word_frequencies = Counter(chain(*self.captions)).most_common()
        words = self.word_frequencies[:, 0]
        self.word_to_id = {self.PAD:0, self.BOS:1, self.EOS:2}
        self.word_to_id.update({word:word_id for word_id, word
                                in enumerate(words, 3)})
        self.id_to_word = {word_id:word for word, word_id
                                in self.word_to_id.items()}
    # 학습 모델 로드
    def load_custom_model(self, train_data_path, model_file_path):
        from Attention import Attention
        from models import NIC
        from generator import Generator

        data_logs = np.genfromtxt(target_data_path + 'data_parameters.log', delimiter=' ', dtype='str')
        data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))

        MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
        IMG_FEATS = int(data_logs['IMG_FEATS:'])
        VOCABULARY_SIZE = len(self.word_to_id)

        self.model =  NIC(max_token_length=MAX_TOKEN_LENGTH,
                vocabulary_size=VOCABULARY_SIZE,
                tweet_max_len=self.tweet_max_len,
                tweet_max_words=self.tweet_max_words,
                rnn='gru',
                num_image_features=IMG_FEATS,
                hidden_size=256,
                embedding_size=128,
                embedding_weights=None)
        
        self.model.load_weights(model_file_path)
        
    # 예측에 사용할 parameter 정보 로드
    def _load_log_file(self):
        data_logs = np.genfromtxt(self.target_data_path + 'data_parameters.log',delimiter=' ', dtype='str')
        data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))
        return data_logs

    # Tokenizer 에 target_csv 의 사용자 입력 텍스트(tweets) fit 시키는 작업
    def tweets_to_token(self):
        tweets = self.target_csv['tweets'].values
        self.tokenizer = Tokenizer(num_words=self.tweet_max_words, lower=True)
        self.tokenizer.fit_on_texts(tweets)

    
        
    # 각 이미지, 텍스트 활용 예측
    def predict(self, hashtag_cnt = 12, include_tweet = True, include_img = True): 
        # image_names = self.target_csv['image_names'].tolist()
        image_names = list(self.image_names_to_features.keys())
        tweet_list = self.target_csv['tweets'].tolist()
        count=0

        total_list = []

        for image_arg,image_name in tqdm(enumerate(image_names)):
            count+=1
            temp = [image_name]
            tweet=str(tweet_list[image_arg])
            sequences = self.tokenizer.texts_to_sequences([tweet])
            tweet_vec=pad_sequences(sequences, maxlen=self.tweet_max_len)

            features = self.image_names_to_features[image_name]['image_features'][:]

            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,self.IMG_FEATS))

            image_features[0, 0, :] = features

            neural_caption = []
            num=0
            list_word_id=[]
            
            if(include_tweet and include_img):
                predictions = self.model.predict([text, image_features,tweet_vec])
            elif(include_tweet):
                predictions = self.model.predict([text,tweet_vec])
            elif(include_img):
                predictions = self.model.predict([text,image_features])
            
            for word_arg in range(self.MAX_TOKEN_LENGTH-1):
                matrix=np.argsort(predictions[0, word_arg, :])
                for id in reversed(matrix):
                    if id not in list_word_id:
                        target_word_id=id
                        list_word_id.append(id)
                        break
                
                next_word_arg = word_arg + 1
                text[0, next_word_arg, target_word_id] = 1
                word = self.id_to_word[target_word_id]
                
                num+=1
                if word == '<E>':
                    break
                elif(num==(hashtag_cnt+1)):
                    break
                else:
                    neural_caption.append(word)

            temp += neural_caption
            total_list.append(temp)

        pred_df = pd.DataFrame(total_list)

        return pred_df

# 학습 데이터 파일
train_data_path = './dataset/preprocessed_data/'                    # 학습에 사용된 데이터
word_to_id_filename = 'word_to_id.p'                                # 학습에 사용된 해시태그 정보 파일 이름
id_to_word_filename = 'id_to_word.p'                                # 학습에 사용된 해시태그 정보
model_file_path = './trained_models/image_text/weights_final.hdf5'  # 학습 모델 이름

# 해시태그 예측 대상 파일
extract_img_feature = False                                             # 이미지 피처 추출 수행 여부
cnn_extractor = 'resnet50'                                              # 이미지 피처 추출 모델
target_data_path = './test_dataset/preprocessed_data/'                  # 예측할 이미지와 이미지 피처가 저장된 경로
target_csv_path = './test_dataset/image_text_tag.csv'                   # 예측할 게시글에 대한 csv 경로
target_img_path = './test_dataset/'                                     # 예측할 이미지 경로
image_name_to_features_filename = 'resnet50_image_name_to_features.h5'  # 예측할 이미지 피처

# 모델 호출
pred_model = Predict_AMNN(train_data_path, 
                          extract_img_feature, 
                          target_img_path, 
                          cnn_extractor, 
                          target_data_path, 
                          target_csv_path , 
                          word_to_id_filename, 
                          id_to_word_filename, 
                          image_name_to_features_filename,
                          model_file_path
                          )

# 출력할 해시태그 수, tweets 포함 여부, img 포함 여부
hashtag_cnt = 20
include_tweet = True
include_img = True

# 해시태그 예측
pred = pred_model.predict(hashtag_cnt, include_tweet, include_img)

columns = ['name']
columns += [f'hashtag_{i}' for i in range(1,pred.shape[1])]
pred.columns = columns

pred.to_csv(f"predict_hashtag_{hashtag_cnt}.csv",index=False,encoding='utf-8-sig')
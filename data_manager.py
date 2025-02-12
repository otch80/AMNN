from collections import Counter
from itertools import chain
import os
import pickle
from string import digits
import time
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py
import numpy as np
import pandas as pd
import cv2


class DataManager(object):
    """Data manegment and pre-preprocessor class
    # Arguments
        data_filename: File which contains in every row the caption and
            the image name, separated by the character given in sep.
        extract_image_features: Flag to create a h5py file that
            contains a vector of features extracted by a pre-trained
            CNN given in cnn_extractor.
        image_directory: Path to the images for which we will extract
            their features.
    """

    def __init__(self, data_filename, max_caption_length=20, sep='*',
                 word_frequency_threshold=2, randomize_data=True,
                 split_data=True, extract_image_features=False,
                 image_directory=None, dump_path='preprocessed_data',
                 cnn_extractor='inception'):

        self.data_filename = data_filename
        self.max_caption_length = max_caption_length
        self.sep = sep
        self.word_frequency_treshold = word_frequency_threshold
        self.randomize_data = randomize_data
        self.split_data_flag = split_data
        self.extract_image_features = extract_image_features
        self.image_directory = image_directory
        self.dump_path = dump_path
        self.cnn_extractor = cnn_extractor
        if self.cnn_extractor == 'inception':
            self.IMG_FEATS = 2048
        elif self.cnn_extractor == 'vgg16':
            self.IMG_FEATS = 4096
        elif self.cnn_extractor == 'vgg19':
            self.IMG_FEATS = 4096
        elif self.cnn_extractor == 'vgg16_places':
            self.IMG_FEATS = 4096
        elif self.cnn_extractor == 'inceptionresnetv2':
            self.IMG_FEATS = 1536
        elif self.cnn_extractor == 'resnet50':
            self.IMG_FEATS = 2048
        elif self.cnn_extractor == 'xception':
            self.IMG_FEATS = 2048
        elif self.cnn_extractor== 'resnet152':
            self.IMG_FEATS = 2048
        else:
            raise Exception('Invalid CNN name')

        self.original_directory = os.getcwd()
        self.BOS = '<S>' #Beginning Of Sentence
        self.EOS = '<E>' #End Of Sentence
        self.PAD = '<P>'
        self.word_frequencies = None
        self.captions = None
        self.image_files = None
        self.image_features = None
        self.word_to_id = None
        self.id_to_word = None
        self.extracted_features = None
        self.features_file_names = None
        self.image_feature_files = None
        self.elapsed_time = None

        if self.extract_image_features == True:
            assert self.image_directory != None

    def preprocess(self):
        start_time = time.monotonic()
        self.load(self.data_filename)
        self.remove_long_captions()
        self.get_corpus_statistics()
        self.remove_infrequent_words()
        self.construct_dictionaries()
        if self.extract_image_features == True:
            self.get_image_features(self.image_directory)
            self.move_to_path()
            self.write_image_features_to_h5()
            self.move_path_back()
        self.move_to_path()
        self.write_data()
        self.write_dictionaries()
        self.elapsed_time = time.monotonic() - start_time
        self.write_parameters()
        if self.split_data_flag == True:
            self.split_data()
        self.move_path_back()

    def load(self, data_filename):

        print('Loading data ...')
        try:
          data = pd.read_table(data_filename, sep=self.sep)
        except:
          data = pd.read_csv(data_filename, on_bad_lines='skip') # 추가 - 예외

        data = np.asarray(data)
        if self.randomize_data == True:
            np.random.shuffle(data)
        self.image_files = data[:, 0]
        self.captions = data[:, 2]
        self.tweets=data[:,1]
        number_of_captions = self.image_files.shape[0]
        print('Loaded', number_of_captions, 'captions')

    def remove_long_captions(self):
        print('Removing captions longer than', self.max_caption_length, '...')
        reduced_image_files = []
        reduced_tweets=[]
        reduced_captions = []
        previous_file_size = len(self.captions)
        for image_arg, caption in enumerate(self.captions):
            lemmatized_caption = self.lemmatize_sentence(caption)
            if (len(lemmatized_caption) <= self.max_caption_length):
                reduced_captions.append(lemmatized_caption)
                reduced_tweets.append(self.tweets[image_arg])
                reduced_image_files.append(self.image_files[image_arg])

        self.captions = reduced_captions
        self.image_files = reduced_image_files
        self.tweets=reduced_tweets
        current_file_size = len(self.captions)
        file_difference = previous_file_size - current_file_size
        print('Number of files removed:', file_difference)
        print('Current number of files:', current_file_size)
        self.initial_number_of_captions = previous_file_size
        self.number_of_captions_removed = file_difference
        self.current_number_of_captions = current_file_size

    def lemmatize_sentence(self, caption):
        incorrect_chars = digits + ";.,'/*?¿><:{}[\]|+"
        char_translator = str.maketrans('', '', incorrect_chars)
        quotes_translator = str.maketrans('', '', '"')
        clean_caption = caption.strip().lower()
        clean_caption = clean_caption.translate(char_translator)
        clean_caption = clean_caption.translate(quotes_translator)
        clean_caption = clean_caption.split(' ')
        return clean_caption

    def get_corpus_statistics(self):
        self.word_frequencies = Counter(chain(*self.captions)).most_common()

    def remove_infrequent_words(self):
        # TODO Add option to remove captions that have a words not in vocabulary
        print('Removing words with a frequency less than',
                        self.word_frequency_treshold,'...')
        frequent_threshold_arg=len(self.word_frequencies)  # set default frequent_threshold_arg
        for frequency_arg, frequency_data in enumerate(self.word_frequencies):
            frequency = frequency_data[1]
            if frequency <= self.word_frequency_treshold:
                frequent_threshold_arg = frequency_arg
                break

        previous_vocabulary_size = len(self.word_frequencies)
        if self.word_frequency_treshold != 0:
            self.word_frequencies = np.asarray(
                        self.word_frequencies[0:frequent_threshold_arg])
        else:
            self.word_frequencies = np.asarray(self.word_frequencies)

        current_vocabulary_size = self.word_frequencies.shape[0]
        vocabulary_difference = (previous_vocabulary_size -
                                current_vocabulary_size)
        print('Number of words removed:',vocabulary_difference)
        print('Current number of words:',current_vocabulary_size)

        self.initial_number_of_words = previous_vocabulary_size
        self.number_of_words_removed = vocabulary_difference
        self.current_number_of_words = current_vocabulary_size

    def construct_dictionaries(self):
        words = self.word_frequencies[:, 0]
        self.word_to_id = {self.PAD:0, self.BOS:1, self.EOS:2}
        self.word_to_id.update({word:word_id for word_id, word
                                in enumerate(words, 3)})
        self.id_to_word = {word_id:word for word, word_id
                                in self.word_to_id.items()}

    def get_image_features(self, image_directory):

        from keras.preprocessing import image
        from keras.models import Model

        if self.cnn_extractor == 'vgg16':

            from keras.applications.vgg16 import preprocess_input
            from keras.applications import VGG16

            self.IMG_FEATS = 4096
            base_model = VGG16(weights='imagenet')
            model =  Model(inputs=base_model.input,
                            outputs=base_model.get_layer('fc2').output)
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in tqdm(enumerate(self.image_feature_files)):
                image_path = image_directory + image_file
                img = image.load_img(image_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)

        elif self.cnn_extractor == 'vgg19':

            from keras.applications.vgg19 import preprocess_input
            from keras.applications import VGG19

            self.IMG_FEATS = 4096
            base_model = VGG19(weights='imagenet')
            model =  Model(inputs=base_model.input,
                            outputs=base_model.get_layer('fc2').output)
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in enumerate(self.image_feature_files):
                image_path = image_directory + image_file
                if image_arg%100 == 0:
                    print('%.2f %% completed' %
                            round(100*image_arg/number_of_images,2))
                img = image.load_img(image_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)

        elif self.cnn_extractor == 'inception':

            from keras.applications.inception_v3 import preprocess_input
            from keras.applications import InceptionV3

            self.IMG_FEATS = 2048
            base_model = InceptionV3(weights='imagenet')
            model =  Model(inputs=base_model.input,
                                outputs=base_model.get_layer('avg_pool').output)
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in tqdm(enumerate(self.image_feature_files)):
                image_path = image_directory + image_file
                img = image.load_img(image_path, target_size=(299, 299))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)
        elif self.cnn_extractor == 'vgg16_places':
            from VGG_Place205 import VGG16_places
            from keras.applications.vgg16 import preprocess_input
            self.IMG_FEATS = 4096
            base_model = VGG16_places()
            base_model.load_weights("VGG_Place205/vgg_tensorflow_channels_last.h5")
            model =  Model(inputs=base_model.input,outputs=base_model.get_layer('fc7').output)
            
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in enumerate(self.image_feature_files):
                image_path = image_directory + image_file
                if image_arg%100 == 0:
                    print('%.2f %% completed' %
                            round(100*image_arg/number_of_images,2))
                img = image.load_img(image_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)
        elif self.cnn_extractor=='inceptionresnetv2':
            from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
            base_model = InceptionResNetV2(weights="imagenet")
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in enumerate(self.image_feature_files):
                image_path = image_directory + image_file
                if image_arg%100 == 0:
                    print('%.2f %% completed' %
                            round(100*image_arg/number_of_images,2))
                img = image.load_img(image_path, target_size=(229, 229))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)
        elif self.cnn_extractor=='resnet50':
            # from keras.applications.resnet50 import ResNet50, preprocess_input
            from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # 추가 - 변경
            base_model = ResNet50(weights="imagenet")
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in tqdm(enumerate(self.image_feature_files)):
                
                if ".jpg" in image_file:
                    image_path = image_directory + image_file
                else:
                    image_path = str(image_directory) + str(image_file) + ".jpg" # 추가 - 확장자
                img = image.load_img(image_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))

            self.extracted_features = np.asarray(self.extracted_features)
        elif self.cnn_extractor=='xception':
            from keras.applications.xception import Xception, preprocess_input
            base_model = Xception(weights="imagenet")
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in enumerate(self.image_feature_files):
                image_path = image_directory + image_file
                if image_arg%100 == 0:
                    print('%.2f %% completed' %
                            round(100*image_arg/number_of_images,2))
                img = image.load_img(image_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)
        elif self.cnn_extractor=='resnet152':
            from resnet152 import resnet152_model
            base_model = resnet152_model("resnet152_weights_tf.h5")
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.extracted_features = []
            self.image_feature_files = list(set(self.image_files))
            number_of_images = len(self.image_feature_files)
            for image_arg,image_file in enumerate(self.image_feature_files):
                image_path = image_directory + image_file
                if image_arg%100 == 0:
                    print('%.2f %% completed' %
                            round(100*image_arg/number_of_images,2))
                img = cv2.resize(cv2.imread(image_path), (224, 224)).astype(np.float32)
                # Remove train image mean
                img[:,:,0] -= 103.939
                img[:,:,1] -= 116.779
                img[:,:,2] -= 123.68
                img = np.expand_dims(img, axis=0)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)
            
    def write_image_features_to_h5(self):
        print('Writing image features to h5...')
        try:
          dataset_file = h5py.File(self.cnn_extractor +
                                 '_image_name_to_features.h5')
        except: # 추가 - 예외
          dataset_file = h5py.File(self.cnn_extractor +
                                 '_image_name_to_features.h5','w')
        number_of_features = len(self.image_feature_files)
        for image_arg, image_file in tqdm(enumerate(self.image_feature_files)):
            
            if not type(image_file) is str:
                file_id = dataset_file.create_group(str(image_file)) # 추가 - 타입체크
            else:
                file_id = dataset_file.create_group(image_file)

            image_data = file_id.create_dataset('image_features',(self.IMG_FEATS,), dtype='float32')
            image_data[:] = self.extracted_features[image_arg,:]

        dataset_file.close()

    def write_image_feature_files(self):
        pickle.dump(self.image_feature_files,
                    open('image_feature_files.p', 'wb'))

    def write_dictionaries(self):
        pickle.dump(self.word_to_id, open('word_to_id.p', 'wb'))
        pickle.dump(self.id_to_word, open('id_to_word.p', 'wb'))

    def write_image_features(self):
        pickle.dump(self.extracted_features,
                    open('extracted_features.p', 'wb'))

    def write_parameters(self):
        log_file = open('data_parameters.log','w')
        log_file.write('data_filename %s \n' %self.data_filename)
        log_file.write('dump_path %s \n' %self.dump_path)
        log_file.write('BOS: %s \n' % self.BOS)
        log_file.write('EOS: %s \n' % self.EOS)
        log_file.write('PAD: %s \n' % self.PAD)
        log_file.write('IMG_FEATS: %s \n' %self.IMG_FEATS)
        log_file.write('word_frequency_threshold: %s \n'
                        %self.word_frequency_treshold)
        log_file.write('max_caption_length: %s \n'
                        %self.max_caption_length)
        log_file.write('initial_data_size: %s \n'
                        %self.initial_number_of_captions)
        log_file.write('captions_larger_than_threshold: %s \n'
                        %self.number_of_captions_removed)
        log_file.write('current_data_size: %s \n'
                        %self.current_number_of_captions)
        log_file.write('initial_word_size: %s \n'
                        %self.initial_number_of_words)
        log_file.write('words_removed_by_frequency_threshold %s \n'
                        %self.number_of_words_removed)
        log_file.write('current_word_size: %s \n'
                        %self.current_number_of_words)
        log_file.write('cnn_extractor: %s \n' %self.cnn_extractor)
        log_file.write('elapsed_time: %s' %self.elapsed_time)
        log_file.close()

    def write_data(self):
        data_file = open('complete_data.txt','w')
        data_file.write('image_names*tweets*hashtags\n')
        for image_arg, image_name in enumerate(self.image_files):
            caption = ' '.join(self.captions[image_arg])
            data_file.write('%s*%s*%s\n' %(image_name, self.tweets[image_arg],caption))
        data_file.close()

    def move_to_path(self):
        directory = self.dump_path
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)

    def move_path_back(self):
        os.chdir(self.original_directory)

    def split_data(self, train_porcentage=.90):

        try:
          complete_data = pd.read_table('complete_data.txt',sep='*')
        except:
          complete_data = pd.read_table('complete_data.txt',sep='*', on_bad_lines='skip') # 추가 - 예외

        data_size = complete_data.shape[0]
        training_size = int(data_size*1)
        complete_training_data = complete_data[0:training_size]
        test_data = complete_data[training_size:]
        test_data.to_csv('test_data.txt',sep='*',index=False)
        # splitting between validation and training 
        training_size = int(training_size*train_porcentage)
        validation_data = complete_training_data[training_size:]
        training_data = complete_training_data[0:training_size]
        validation_data.to_csv('validation_data.txt',sep='*',index=False)
        training_data.to_csv('training_data.txt',sep='*',index=False)
        print('num of training data size:',training_size)
        print('num of validation data size:',len(complete_training_data)-training_size)


if __name__ == '__main__':

    root_path = '../../datasets/image_text/'
    captions_filename = root_path + 'image_text_data.txt'
    image_path='/home/eric/data/social_images/'

    data_manager = DataManager(data_filename = captions_filename,
                                max_caption_length = 30,
                                word_frequency_threshold = 1,
                                extract_image_features = True,
                                image_directory = image_path,
                                cnn_extractor = 'inception',
                                split_data = True,
                                dump_path =root_path+'preprocessed_data/')

    data_manager.preprocess()
    print(data_manager.captions[0])
    print(data_manager.word_frequencies[0:30])
    # data_manager.write_image_features_to_h5()

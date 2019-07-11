from pathlib import Path
import cv2
import dlib
import numpy as np
from keras.utils.data_utils import get_file
from age_model import get_model

class PredictAge():

    def __init__(self):
        # self.pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/age_only_resnet50_weights.061-3.300-4.410.hdf5"
        # self.modhash = "306e44200d3f632a5dccac153c2966f2"
        self.model_name = "ResNet50"
        self.margin = 0.4
        #self.weight_file = get_file("age_only_resnet50_weights.061-3.300-4.410.hdf5", self.pretrained_model,
                           # cache_subdir="age_pretrained_models",
                           # file_hash=self.modhash, cache_dir=Path(__file__).resolve().parent)
        self.weight_file = "age_pretrained_models//age_only_resnet50_weights.061-3.300-4.410.hdf5"
        self.detector = dlib.get_frontal_face_detector()

        # load model and weights
        self.model = get_model(model_name=self.model_name)
        self.model.load_weights(self.weight_file)
        self.img_size = self.model.input.shape.as_list()[1]

    def predict_age(self, img_path=None, img=None):
        print(img_path)
        print(img)
        #img_path = "C:/Users/11154/Desktop/age-gender-estimation-master/age-gender-estimation-master/data/wiki_crop/00/1335100_1954-09-18_2007.jpg"
        # model_name = "ResNet50"
        # margin = 0.4
        # weight_file = get_file("age_only_resnet50_weights.061-3.300-4.410.hdf5", pretrained_model,
        #                            cache_subdir="age_pretrained_models",
        #                            file_hash=modhash, cache_dir=Path(__file__).resolve().parent)
        # for face detection
        #detector = dlib.get_frontal_face_detector()

        # load model and weights
        #model = get_model(model_name=model_name)
        #model.load_weights(weight_file)
        #img_size = model.input.shape.as_list()[1]
        if img_path is not None:
            img = cv2.imread(str(img_path), 1)
        #print(img)
        h, w, _ = img.shape
        print(h,w)
        r = 640 / max(w, h)
        img = cv2.resize(img, (int(w * r), int(h * r)))

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        print(img_h, img_w)
        # detect faces using dlib detector
        detected = self.detector(input_img, 1)
        print(detected)
        faces = np.empty((len(detected), self.img_size, self.img_size, 3))
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - self.margin * w), 0)
                yw1 = max(int(y1 - self.margin * h), 0)
                xw2 = min(int(x2 + self.margin * w), img_w - 1)
                yw2 = min(int(y2 + self.margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.img_size, self.img_size))

            # predict ages and genders of the detected faces
            results = self.model.predict(faces)
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results.dot(ages).flatten()
            print(predicted_ages)
            for i, d in enumerate(detected):
                label = str(int(predicted_ages[i]))
                return label


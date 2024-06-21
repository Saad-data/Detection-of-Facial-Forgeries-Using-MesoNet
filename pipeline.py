import random
from os import listdir
from os.path import isfile, join
import numpy as np
from math import floor
from scipy.ndimage import zoom, rotate
import face_recognition

# Class for handling image data
class ImageData:
    def __init__(self, path):
        self.path = path
        self.frames = self.load_images(path)
        self.length = len(self.frames)

    def load_images(self, path):
        # Load images from the directory
        return [face_recognition.load_image_file(join(path, f)) for f in listdir(path) if isfile(join(path, f))]

    def get_frame(self, index):
        return self.frames[index]

# Class to find faces in images
class FaceFinder(ImageData):
    def __init__(self, path, load_first_face=True):
        super().__init__(path)
        self.faces = {}
        self.coordinates = {}
        self.last_frame = self.get_frame(0)
        self.frame_shape = self.last_frame.shape[:2]
        self.last_location = (0, 200, 200, 0)
        if load_first_face:
            face_positions = face_recognition.face_locations(self.last_frame, number_of_times_to_upsample=2)
            if face_positions:
                self.last_location = face_positions[0]

    def expand_location(self, loc, margin=0.2):
        offset = round(margin * (loc[2] - loc[0]))
        y0 = max(loc[0] - offset, 0)
        x1 = min(loc[1] + offset, self.frame_shape[1])
        y1 = min(loc[2] + offset, self.frame_shape[0])
        x0 = max(loc[3] - offset, 0)
        return (y0, x1, y1, x0)

    @staticmethod
    def upsample_location(reduced_loc, origin, factor):
        y0, x1, y1, x0 = reduced_loc
        return (
            round(origin[0] + y0 * factor),
            round(origin[1] + x1 * factor),
            round(origin[0] + y1 * factor),
            round(origin[1] + x0 * factor),
        )

    @staticmethod
    def find_largest_location(locations):
        return max(locations, key=lambda loc: loc[2] - loc[0])

    def find_faces(self, resize=0.5):
        for i in range(self.length):
            frame = self.get_frame(i)
            potential_loc = self.expand_location(self.last_location)
            face_patch = frame[potential_loc[0]:potential_loc[2], potential_loc[3]:potential_loc[1]]
            face_patch_small = zoom(face_patch, (resize, resize, 1))
            reduced_locs = face_recognition.face_locations(face_patch_small, model='cnn')

            if reduced_locs:
                largest_loc = self.find_largest_location(reduced_locs)
                upsampled_loc = self.upsample_location(largest_loc, potential_loc[:2], 1 / resize)
                self.faces[i] = upsampled_loc
                self.last_location = upsampled_loc
                landmarks = face_recognition.face_landmarks(frame, [upsampled_loc])
                if landmarks:
                    self.coordinates[i] = self.extract_coordinates(landmarks[0])

    def extract_coordinates(self, landmark, K=2.2):
        E1 = np.mean(landmark['left_eye'], axis=0)
        E2 = np.mean(landmark['right_eye'], axis=0)
        E = (E1 + E2) / 2
        N = np.mean(landmark['nose_tip'], axis=0) / 2 + np.mean(landmark['nose_bridge'], axis=0) / 2
        B1 = np.mean(landmark['top_lip'], axis=0)
        B2 = np.mean(landmark['bottom_lip'], axis=0)
        B = (B1 + B2) / 2
        C = N
        l1 = np.linalg.norm(E1 - E2)
        l2 = np.linalg.norm(B - E)
        l = max(l1, l2) * K
        rot = np.arctan2(B[0] - E[0], B[1] - E[1]) * 180 / np.pi
        return ((floor(C[1]), floor(C[0])), floor(l), rot)

    def get_aligned_face(self, i, l_factor=1.3):
        frame = self.get_frame(i)
        if i in self.coordinates:
            c, l, r = self.coordinates[i]
            l = int(l * l_factor)
            dl_ = floor(np.sqrt(2) * l / 2)
            patch = frame[floor(c[0] - dl_):floor(c[0] + dl_), floor(c[1] - dl_):floor(c[1] + dl_)]
            rotated_patch = rotate(patch, -r, reshape=False)
            return rotated_patch[floor(dl_ - l // 2):floor(dl_ + l // 2), floor(dl_ - l // 2):floor(dl_ + l // 2)]
        return frame

# Class to generate batches of face images from frames
class FaceBatchGenerator:
    def __init__(self, face_finder, target_size=256):
        self.finder = face_finder
        self.target_size = target_size
        self.head = 0

    def resize_patch(self, patch):
        m, n = patch.shape[:2]
        return zoom(patch, (self.target_size / m, self.target_size / n, 1))

    def next_batch(self, batch_size=50):
        batch = []
        while len(batch) < batch_size and self.head < len(self.finder.coordinates):
            if self.head in self.finder.coordinates:
                patch = self.finder.get_aligned_face(self.head)
                batch.append(self.resize_patch(patch))
            self.head += 1
        return np.array(batch)

def predict_faces(generator, classifier, batch_size=50):
    profile = []
    while generator.head < len(generator.finder.coordinates):
        face_batch = generator.next_batch(batch_size=batch_size)
        if face_batch.size > 0:
            prediction = classifier.predict(face_batch)
            profile.extend(prediction)
    return np.array(profile)

def compute_accuracy(classifier, dirname, frame_subsample_count=30):
    filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and f.endswith(('.jpg', '.png'))]
    predictions = {}
    for img in filenames:
        print(f'Processing image {img}')
        face_finder = FaceFinder(join(dirname, img), load_first_face=False)
        face_finder.find_faces(resize=0.5)
        print(f'Predicting {img}')
        gen = FaceBatchGenerator(face_finder)
        p = predict_faces(gen, classifier)
        predictions[img] = (np.mean(p > 0.5), p)
    return predictions

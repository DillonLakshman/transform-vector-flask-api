from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

import subprocess
import imghdr
import csv
import os

import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.models import Sequential, load_model
from sklearn.cluster import MiniBatchKMeans
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
CORS(app)


class Converter:
    def convert(self):
        file_path = request.args.get('file_path')
        file_name = request.args.get('file_name')
        qtres = request.args.get('qtres')
        ltres = request.args.get('ltres')
        pathomit = request.args.get('pathomit')
        k = request.args.get('k')
        denoise_bool = request.args.get('denoiseBool')
        end_file_type = ".svg"

        image_extension = imghdr.what(file_path)

        convertImg = cv2.imread(file_path)
        convertImg = Analyzer.image_resize(convertImg, 1500)
        convertImg = Analyzer.quantize_image(convertImg, int(k))

        if denoise_bool:
            convertImg = cv2.fastNlMeansDenoisingColored(convertImg, None, 10, 10, 7, 21)

        Analyzer.save_image_to_path(os.path.join("temp", "convert", "temp", "temp."+image_extension), convertImg)

        subprocess.call([
            'java', '-jar', 'imageTrace.jar', os.path.join("temp", "convert", "temp", "temp."+image_extension),
            'outfilename', file_path.split(".")[0]+end_file_type,
            'colorsampling', str(0),
            'numberofcolors', str(k),
            'ltres', str(ltres),
            'qtres', str(qtres),
            'pathomit', str(pathomit)
        ])

        return jsonify(progress="completed", file_name=file_name, file_path=file_path
                       , out_path=file_path.split(".")[0]+end_file_type)


class Analyzer:
    def prepare_image(path: str):
        img_size = 250
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized_array = cv2.resize(img_array, (img_size, img_size))
        return resized_array.reshape(-1, img_size, img_size, 1)

    # return number of colors in image
    def count_colours(src: str):
        image = cv2.imread(src)
        unique, counts = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
        return counts.size

    def classify(path: str):
        my_model = load_model('classification_model/gis_classify.h5')
        prediction = my_model.predict(Analyzer.prepare_image(path))

        # ["LandClassificationMaps", "Satellite", "Scanned"]

        values = prediction[0]
        type = "unidentified"
        category= "unidentified"

        if values[0] == 1:
            type = "Land Classification Maps"
            category = "landclass"
        elif values[1] == 1:
            type = "Satellite Imagery"
            category = "sat"
        elif values[2] == 1:
            type = "Scanned Map"
            category = "scanned"

        return [type, category]

    def identify_quntize_cluster(colors: int):
        if colors < (2 ** 4):
            return 4
        elif colors < (2 ** 8):
            return 8
        elif colors < (2 ** 12):
            return 12
        elif colors < (2 ** 16):
            return 16
        elif colors < (2 ** 20):
            return 20
        elif colors < (2 ** 24):
            return 24
        elif colors < (2 ** 28):
            return 28
        else:
            return 32

    def get_best_param_range(self, category):
        path = 'final_csv/'+category+'.csv'

        ltres_low = Analyzer.find_lowest(path, "ltres")
        ltres_high = Analyzer.find_highest(path, "ltres")

        qtres_low = Analyzer.find_lowest(path, "qtres")
        qtres_high = Analyzer.find_highest(path, "qtres")

        pathomit_low = Analyzer.find_lowest(path, "pathomit")
        pathomit_high = Analyzer.find_highest(path, "pathomit")

        print(ltres_low, ltres_high, qtres_low, qtres_high, pathomit_low, pathomit_high)

        return [('ltres_low', ltres_low), ('ltres_high', ltres_high), ('qtres_low', qtres_low),
                ('qtres_high', qtres_high),
                ('pathomit_low', pathomit_low), ('pathomit_high', pathomit_high)]

    def find_lowest(path, prop_val):
        input_file = csv.DictReader(open(path))
        lowest = 0
        index = 0

        for row in input_file:
            if index == 0:
                lowest = row[prop_val]
            else:
                if row[prop_val] < lowest:
                    lowest = row[prop_val]

            index += 1

        return lowest

    def find_highest(path, prop_val):
        input_file = csv.DictReader(open(path))
        highest = 0
        index = 0

        for row in input_file:
            if index == 0:
                highest = row[prop_val]
            else:
                if row[prop_val] > highest:
                    highest = row[prop_val]

            index += 1

        return highest

    # resize Image
    def image_resize(image_to_resize: object, set_width: int):
        width = int(image_to_resize.shape[1] * set_width / image_to_resize.shape[1])
        height = int(image_to_resize.shape[0] * set_width / image_to_resize.shape[1])
        dimensions = (width, height)
        return cv2.resize(image_to_resize, dimensions, interpolation=cv2.INTER_AREA)

    # quantize image to reduce colour data for easy raster to vector conversion
    def quantize_image(image: object, number_of_colors: int):
        (h, w) = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=int(number_of_colors))
        labels = clt.fit_predict(image)
        quantized_image = clt.cluster_centers_.astype("uint8")[labels]
        quantized_image = quantized_image.reshape((h, w, 3))
        quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)
        # return resized image
        return quantized_image

    def save_image_to_path(path: str, image: object) -> object:
        cv2.imwrite(path, image)

    def convert_to_png(in_path: str, out_path: str):
        subprocess.call(["svg2png.bat", in_path, out_path])

    def check_similarity(path_1: str, path_2: str, image_size):
        img1 = cv2.imread(path_1)
        img2 = Analyzer.image_resize(cv2.imread(path_2), image_size)

        w1, h1 = img2.shape[:-1]
        img1 = cv2.resize(img1, (h1, w1))

        s = ssim(img1, img2, multichannel=True)
        return str(s * 100)

    def get_params(image_path, category):
        try:
            os.rmdir('temp/png')
        except OSError as e:
            print("does not exist")

        try:
            os.rmdir('temp/svg')
        except OSError as e:
            print("does not exist")

        try:
            os.mkdir('temp/svg')
        except OSError as e:
            print("does not exist")

        try:
            os.mkdir('temp/png')
        except OSError as e:
            print("does not exist")

        end_file_type = ".svg"
        param_range = Analyzer().get_best_param_range(category)
        image_size = 120

        image = cv2.imread(image_path)
        image = Analyzer.image_resize(image, image_size)
        image = Analyzer.quantize_image(image, 16)
        image_extension = imghdr.what(image_path)
        temp_image_path = 'temp/sample.' + image_extension
        Analyzer.save_image_to_path(temp_image_path, image)

        index = 0

        with open('temp/temp.csv', 'w', newline='') as f:
            thewriter = csv.writer(f)

            thewriter.writerow(["index", "ltres", "qtres", "pathomit", "file_path", "similarity"])

            for ltres in range(int(param_range[0][1]), int(param_range[1][1]) + 1, 2):
                print(str(param_range[3][1])+"asd")
                for qtres in range(int(param_range[2][1]), int(param_range[3][1]) + 1, 2):
                    for pathomit in range(int(param_range[4][1]), int(param_range[5][1]) + 1):
                        if pathomit == 1 or pathomit == 10 or pathomit == 100:
                            index = index + 1

                            svg_out_path = "temp/svg/" + str(index) + end_file_type

                            subprocess.call([
                                'java', '-jar', 'imageTrace.jar', temp_image_path,
                                'outfilename', svg_out_path,
                                'pathomit', str(1 / pathomit),
                                'ltres', str(ltres),
                                'qtres', str(qtres),
                                'colorsampling', str(0),
                                'colorquantcycles', str(16)
                            ])

                            png_out_path = "temp/png/" + str(index) + ".png"

                            Analyzer.convert_to_png("../../../" + svg_out_path, "../../../" + png_out_path)

                            similarity_val = Analyzer.check_similarity(temp_image_path, png_out_path, image_size)

                            thewriter.writerow(
                                [str(index), str(ltres), str(qtres), str(pathomit), str(index) + end_file_type,
                                 str(similarity_val)])

        df = pd.read_csv('temp/temp.csv', engine='python')
        df = df.sort_values('similarity', ascending=False)
        df.to_csv(os.path.join('temp/temp_sorted.csv'), index=False)

        input_file = csv.DictReader(open('temp/temp_sorted.csv'))

        # get highest accuracy values
        index = 0

        highest_row = []

        for row in input_file:
            if index < 1:
                highest_row = row

            index += 1

        return highest_row


    def analyze(self):
        file_path = request.args.get('file_path')

        classification = Analyzer.classify(file_path)
        color_count = Analyzer.count_colours(file_path)
        quant_val = Analyzer.identify_quntize_cluster(color_count)

        row = Analyzer.get_params(file_path, classification[1])

        return jsonify(file_path=file_path, classification=str(classification[0]), color_count=color_count
                       , quant_val=quant_val, row=row)

converter = Converter()
analyzer = Analyzer()


@app.route('/convert')
def perform_conversion():
    return converter.convert()


@app.route('/analyze')
def perform_classification():
    return analyzer.analyze()


if __name__ == '__main__':
    app.run()

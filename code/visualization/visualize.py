import time
import os
from os import listdir
from os.path import isfile, join
import pickle
import uuid
import sys


def visualize_images(title, image_name_list):
    images = ""
    for image in image_name_list:
        # img_div = open('../visualization/templates/image-div.html', 'r').read()
        img_div = "{ adName: '<div class=\"col-md-5\"><div class=\"card mb-5\">" \
                  "<img class=\"card-img-top\" src=\"<PATH>\" alt=\"\">" \
                  "<div class=\"card-body\"><IMAGE-TITLE></div></div></div>'},"
        path = get_image_path(image)
        img_div = img_div.replace("<PATH>", path)
        img_div = img_div.replace("<IMAGE-TITLE>", str(image))
        images += img_div
    images.strip(",")
    js = open('../visualization/templates/js.html', 'r').read()
    js = js.replace("<jsonObject>", images)
    js = js.replace("<is-query-result>", "false")
    html_file = open('../visualization/templates/webpage.html', 'r').read()
    html_file = html_file.replace("<mwd-js-content>", js)
    html_file = html_file.replace("<task-title>", title)
    unique_filename = str(uuid.uuid4()).split('-')[1]
    try:
        time_str = str(
            sys.argv[0].split('/')[::-1][0].replace(".py",
                                                    "")) + '-' + str(
            time.strftime("%m%d-%H%M%S")) + '-' + unique_filename
    except:
        time_str = str(sys.argv[0]).replace(".py", "") + '-' + str(time.strftime("%m%d-%H%M%S")) + '-' + unique_filename
    op_file = time_str + '.html'
    with open('../visualization/webpages/' + op_file, 'w') as f:
        f.write(html_file)
    return op_file


def visualize_query_results(title, query_title, query_images, result_title, result_images):
    content_json = "{adName:'<div class=\"form-group col-10\">" \
                   "<hr style=\"border-top: 5px double #8c8b8b;\">" \
                   "</div><h1 class=\"jumbotron-heading\">&nbsp;&nbsp;&nbsp;" + \
                   query_title + "</h1><div class=\"form-group col-10\">" \
                                 "<hr style=\"border-top: 5px double #8c8b8b;\"></div>'},";
    for image in query_images:
        # img_div = open('../visualization/templates/image-div.html', 'r').read()
        img_div = "{ adName: '<div class=\"col-md-5\"><div class=\"card mb-5\">" \
                  "<img class=\"card-img-top\" src=\"<PATH>\" alt=\"\">" \
                  "<div class=\"card-body\"><IMAGE-TITLE></div></div></div>'},"
        path = get_image_path(image)
        img_div = img_div.replace("<PATH>", path)
        img_div = img_div.replace("<IMAGE-TITLE>", str(image))
        content_json += img_div
    content_json += "{adName:'<div class=\"form-group col-10\"><hr style=\"border-top: 5px double #8c8b8b;\"></div>" \
                    "<h1 class=\"jumbotron-heading\">&nbsp;&nbsp;&nbsp;" + \
                    result_title + "</h1>" \
                                   "<div class=\"form-group col-10\">" \
                                   "<hr style=\"border-top: 5px double #8c8b8b;\"></div>'},";
    for image in result_images:
        # img_div = open('../visualization/templates/image-div.html', 'r').read()
        img_div = "{ adName: '<div class=\"col-md-5\"><div class=\"card mb-5\">" \
                  "<img class=\"card-img-top\" src=\"<PATH>\" alt=\"\">" \
                  "<div class=\"card-body\"><IMAGE-TITLE></div></div></div>'},"
        path = get_image_path(image)
        img_div = img_div.replace("<PATH>", path)
        img_div = img_div.replace("<IMAGE-TITLE>", str(image))
        content_json += img_div
    content_json.strip(",")
    js = open('../visualization/templates/js.html', 'r').read()
    js = js.replace("<jsonObject>", content_json)
    js = js.replace("<is-query-result>", "true")
    html_file = open('../visualization/templates/webpage.html', 'r').read()
    html_file = html_file.replace("<mwd-js-content>", js)
    html_file = html_file.replace("<task-title>", title)
    unique_filename = str(uuid.uuid4()).split('-')[1]
    time_str = str(sys.argv[0].split('/')[::-1][0].replace(".py",
                                                           "")) + '-' + str(
        time.strftime("%m%d-%H%M%S")) + '-' + unique_filename
    op_file = time_str + '.html'
    # print(sys.argv[0])
    with open('../visualization/webpages/' + op_file, 'w') as f:
        f.write(html_file)
    return op_file


def get_image_path(image):
    try:
        with open('../visualization/pickles/file_to_path_dict.pkl', 'rb') as pkl_file:
            file_to_path_dict = pickle.load(pkl_file)
    except FileNotFoundError:
        file_to_path_dict = prepare_dict("../../devset/img/")
    return file_to_path_dict[image]


def prepare_dict(path):
    sub_dirs = [x[0] for x in os.walk(path)]
    file_to_path_dict = {}
    for d in sub_dirs:
        for f in listdir(d):
            if isfile(join(d, f)) and len(f) > 3:
                try:
                    file_to_path_dict[int((f[:f.index('.')]))] = d + "/" + f
                except:
                    pass
    with open('../../visualization/pickles/file_to_path_dict.pkl', 'wb') as pkl_file:
        pickle.dump(file_to_path_dict, pkl_file)
    return file_to_path_dict

import xml.etree.ElementTree
import pickle
import glob
import os
import random

from visualization.visualize import visualize_images, visualize_query_results


def get_loc_id_mapping():
    e = xml.etree.ElementTree.parse('../devset/devset_topics.xml').getroot()
    moc1 = e.findall('topic')
    location_title = []
    location_id = []
    for moc in moc1:
        for node in moc.getiterator():
            if node.tag == 'title':
                location_title.append(node.text)
            if node.tag == 'number':
                location_id.append(node.text)
    return dict(zip(location_title, location_id))


def visualize_images_from_graph(image_id, graph_pickle):
    with open(graph_pickle, 'rb') as f:
        img_img_graph = pickle.load(f)
    full_list = [image_id]
    img_list = list(img_img_graph.neighbors(image_id))
    full_list.extend(img_list)
    visualize_images("Neighbor images of " + str(image_id), full_list)


if __name__ == "__main__":
    list_of_files = glob.glob('pickles/cache/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print("using file :" + latest_file)
    with open('../visualization/pickles/file_to_path_dict.pkl', 'rb') as f:
        image_to_loc = pickle.load(f)
    rand_list = random.sample(range(0, len(image_to_loc.keys())), 10)
    # Input = [330089856, 167199643, 330056915]
    # OutputSelf = [7381767396, 4439033482, 5251533840, 4882036306, 1490271201, 6279907527]
    # outputnx = [5251533840, 4882036306, 7381767396, 6279907527, 4439033482, 6031579036]
    # visualize_query_results("Task 4", "Input images", Input, "Result of Networkxs", outputnx)

    # visualize_images_from_graph(4822524241, latest_file)
    # visualize_images_from_graph(1442822407, latest_file)
    # visualize_query_results("Test Query and Results", "Query Images 1 2 3 4 5", [52052758, 6955954692],
    #                        "Result images of 1 2 3 4 5", [6700311309, 6700311309])
    # visualize_images("Test", [list(image_to_loc)[10], list(image_to_loc)[952]])
    for index in rand_list:
        visualize_images_from_graph(list(image_to_loc)[index], latest_file)

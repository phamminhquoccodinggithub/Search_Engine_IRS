# from search.models import Document
import os
import re
DIR_PATH = 'dataset/'
topics = sorted(os.listdir(DIR_PATH))

for topic in topics:
    topic_dir = DIR_PATH + topic + '/'
    for file in sorted(os.listdir(topic_dir)):
        path = topic_dir + file
        with open(path, encoding="utf-8") as f:
            contents = f.readlines()
        title = contents[0]
        url = contents[1]
        content = re.sub("[^\\w\\s]", "", " ".join(contents[5:]))
        result = (title, url, content)
        break

print(result)
# topic_dir_1 = DIR_PATH + topics[0] + '/'
# file_1 = sorted(os.listdir(topic_dir_1))[0]
# file_path = topic_dir_1 + file_1
# with open(file_path, encoding="utf-8") as f:
#     contents = f.readlines()
# print(contents[0])
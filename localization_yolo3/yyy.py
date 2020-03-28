import os;

import numpy as np;
cwd=os.getcwd();
yolo_anchor = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326";
fwrite_anchor = open(cwd+'/model_data/yolo_anchors1.txt', 'w');
fwrite_anchor.write(yolo_anchor);
fwrite_anchor.close();
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(cwd+'/model_data/yolo_anchors.txt') as f:
        anchors = f.readline()
        print("----------------------------------------------------")
        print(anchors)
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
def wrong_file_del(path):
    wrong_file = [];
    file_count = 0;
    with open(path) as f_train:
        lines = f_train.readlines();
        for train_path in lines:
            file_count = file_count+1;
            if 'jpg' in train_path[-6:]:
                wrong_file.append(train_path);

    ret = [new_file for new_file in lines if new_file not in wrong_file];
    f_new = open(path,'w');
    ret = "".join(ret);
    f_new.write(ret);






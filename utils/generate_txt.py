import os
path = "/impacs/yuf5/ImageNet/train"
save_path = "/impacs/yuf5/sample_weight/data_txt/ImageNet/"
def get_all_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list 

for root, dirs, files in os.walk(path):
    for i, dir in enumerate(dirs):
        file_path = os.path.join(path, dir)
        file_list = get_all_files_in_directory(file_path)
        save_path_single = os.path.join(save_path, str(i) +".txt")
        with open(save_path_single, "w") as f:
            for line in file_list:
                f.write(line + "  " + str(i) + "\n" )





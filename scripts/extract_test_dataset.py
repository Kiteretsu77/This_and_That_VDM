'''
    Extract the test dataset from the txt file
'''

if __name__ == "__main__":
    txt_path = "match_info_v2.txt"
    store_path = "test_path_v2.txt"
    start_idx = len("/nfs/turbo/jjparkcv-turbo-large/boyangwa/raw/bridge_data_v2/")

    read_file = open(txt_path, "r")
    write_file = open(store_path, "w")
    for line in read_file.readlines():
        test_dataset_path = line.split(' ')[1]
        test_instance = test_dataset_path[start_idx:]

        write_file.write(test_instance)

        
import os
import datetime
import json


class Dataset:
    def __init__(self, root_path):
        self.root_path = root_path

    def create_subject(self, subject_id, name=None, gender=None):
        subject_path = os.path.join(self.root_path, subject_id)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)

        # 存储一般信息
        info_file_path = os.path.join(subject_path, 'info.txt')
        with open(info_file_path, 'w') as info_file:
            info_file.write(f'Name: {name}\n')
            info_file.write(f'Gender: {gender}\n')

    def generate_save_path(self, subject_id):
        subject_path = os.path.join(self.root_path, subject_id)
        if not os.path.exists(subject_path):
            raise Exception(f"Subject {subject_id} does not exist.")

        # 创建以检查日期的子文件夹
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        check_date_path = os.path.join(subject_path, date_str)
        if not os.path.exists(check_date_path):
            os.makedirs(check_date_path)

        # 获取该日期已有的检查次数，以便命名新的检查结果文件夹
        last_check_num = 0
        for folder_name in os.listdir(check_date_path):
            check_folder_path = os.path.join(check_date_path, folder_name)
            if os.path.isdir(check_folder_path) and folder_name.isdigit():
                last_check_num = max(last_check_num, int(folder_name))
        new_check_num = f"{last_check_num+1:03d}"

        # 创建检查结果文件夹
        check_result_path = os.path.join(check_date_path, new_check_num)
        os.makedirs(check_result_path)
        return check_result_path


    def save_patient_data(self, root_path, patient_id=None, name=None, gender=None, age=None, diagnosis=None, description=None, other=None):
        # 检查是否存在对应的 JSON 文件
        json_file_path = os.path.join(root_path, "patient_info.json")
        if os.path.exists(json_file_path):
            # 如果文件存在，读取已有数据
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            # 如果文件不存在，创建新的数据字典
            data = {}

        # 更新或添加新的数据字段
        if patient_id:
            data["patient_id"] = patient_id
        if name:
            data["name"] = name
        if gender:
            data["gender"] = gender
        if age:
            data["age"] = age
        if diagnosis:
            data["diagnosis"] = diagnosis
        if description:
            data["description"] = description
        if other:
            data["other"] = other

        # 保存数据到 JSON 文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)   
            #将 indent 设置为 4，则输出的 JSON 数据将使用 4 个空格来进行缩进。这样可以使 JSON 数据在文本文件中以更易读的方式进行显示
            #将 ensure_ascii 参数设置为 False，并指定文件的编码为 UTF-8。这样，在生成的 JSON 文件中，中文字符将以原始的 UTF-8 编码进行保存，而不是使用 Unicode 转义序列


if __name__ == '__main__':
    # 使用示例
    root_path = "/home/uax/LiMD_example/Robot_arm/Dataset"
    patient_id = "A000002"
    name = "John"
    gender = "Male"
    age = 30
    diagnosis = "甲状腺结节"
    description = "病例描述内容"
    other = '第5次扫查'

    # 创建数据集对象
    dataset = Dataset(root_path)



    # 创建新主题
    dataset.create_subject(patient_id, name, gender)

    # 检查结果存储路径
    check_result_path = dataset.generate_save_path(patient_id)

    dataset.save_patient_data(check_result_path, patient_id, name, gender, age, diagnosis, description, other)

# RNA_pipline全流程+merge图
import argparse
import os
import pandas as pd
import cv2
import tifffile
from subprocess import Popen, PIPE

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", action="store", dest="input", type=str,
                    help="Input data path (CSV file with 'SN' and 'matrix path' columns).")
parser.add_argument("-o", "--output", action="store", dest="output", type=str,
                    help="Output path.")
args = parser.parse_args()

# 提取输入文件和输出路径
input_path = args.input
output_path = args.output

# 读取CSV文件
df = pd.read_csv(input_path)

# 定义合并图像的函数
def merge_images(output_folder, sn):
    # 构造SN对应的图像文件名
    sn_img_file = f"{sn}.tif"
    sn_mask_file = f"{sn}_mask.tif"

    img_path = os.path.join(output_folder, sn_img_file)
    mask_path = os.path.join(output_folder, sn_mask_file)

    # 检查两个文件是否存在
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"图像文件 {sn_img_file} 或 {sn_mask_file} 在 {output_folder}中不存在")
        return
        # 尝试读取原图和mask图像
    try:
        image = tifffile.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"读取图像时发生错误：{e}, 跳过芯片号 {sn}")
        return

    # # 读取原图和mask图像
    # image = tifffile.imread(img_path)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 在img上绘制mask边界（红色）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

    # 保存合并图像
    merge_file = f"{sn}_merge.tif"
    save_path = os.path.join(output_folder, merge_file)
    cv2.imwrite(save_path, image)

# 假设 df 是一个pandas DataFrame，包含需要处理的数据
for index, row in df.iterrows():
    sn = row['SN']
    matrix_path = row['matrix path']
    output_dir = os.path.join(output_path, sn)
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在处理芯片号: {sn}")
    print(f"芯片号对应矩阵路径: {matrix_path}")

    # 构造命令
    command1 = f"/storeData/USER/data/01.CellBin/00.user/zhangying7/software/Annaconda/envs/python38/bin/python /storeData/USER/data/01.CellBin/00.user/qiaojiao/work_path/SAW_V8/demo/scripts/Dev_rgister/register/main.py -o {output_dir} -v {matrix_path} -w True -wt False --sn {sn} --gpu 0 -RS True -IS False"

    # 执行命令
    os.system(command1)

    # 定义转换 CSV 文件为 XLSX 文件的命令
    python_path = "/storeData/USER/data/01.CellBin/00.user/fengning/01.code/SAW/SAW_version/saw-v8.0.0a17/anaconda/bin/python"
    convert_script = f"import pandas as pd; df = pd.read_csv('{input_path}'); df.to_excel('{os.path.splitext(input_path)[0] + '.xlsx'}', index=False, header=True)"

    # 使用指定的 Python 环境执行转换命令
    convert_command = f"{python_path} -c \"{convert_script}\""

    # 执行命令
    os.system(convert_command)

    # 构造并执行 command2
    # 确保使用更新后的 XLSX 文件路径
    xlsx_input_path = os.path.splitext(input_path)[0] + '.xlsx'
    command2 = f"{python_path} /storeData/USER/data/01.CellBin/00.user/fengning/01.code/SAW/Script/batch_run_gef_index.py -e {xlsx_input_path} -o {output_path} -f {os.path.join(output_path, f'{sn}_error.csv')}"

    # 执行命令
    os.system(command2)

    # 合并图像
    try:
        merge_images(output_dir, sn)
        print(f"芯片号 {sn} 的图像已合并并保存到 {output_dir}")
    except Exception as e:
        print(f"合并图像时发生错误：{e}, 跳过芯片号 {sn}")

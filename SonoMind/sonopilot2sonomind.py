import os
import re
import json
import time
import shutil
import sys
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= 修正跨目录导入路径 =================
# 获取当前脚本所在目录（IS-MAN/SonoMind/）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将父目录（IS-MAN/）添加到Python路径，以便导入SonoPilot模块
sys.path.append(os.path.join(current_dir, '..'))

try:
    # 从同目录导入main_UI.py
    import main_UI
    # 从SonoPilot/RobotServer导入config.py
    from SonoPilot.RobotServer import config
except ImportError as e:
    raise ImportError(
        f"导入配置文件失败，请检查目录结构：{e}\n"
        "当前脚本应位于 IS-MAN/SonoMind/ 目录下\n"
        "config.py 应位于 IS-MAN/SonoPilot/RobotServer/ 目录下\n"
        "main_UI.py 应位于 IS-MAN/SonoMind/ 目录下"
    )

# ================= 配置区域（从配置文件读取） =================
# 监控的根目录 (从 SonoPilot/RobotServer/config.py 导入)
WATCH_DIRECTORY = config.rocord_save_path  # 保留原变量名拼写 rocord
# 输出的目标根目录 (从同目录 main_UI.py 导入)
OUTPUT_ROOT = main_UI.save_root
# 病人文件夹名 (从同目录 main_UI.py 导入)
PATIENT_ID = main_UI.patient_id
# 防抖等待时间（秒）：防止文件还在写入时就触发读取
DEBOUNCE_TIME = 1.5  # 略微增加到1.5秒，适配大文件写入


# ============================================
class NoduleHandler(FileSystemEventHandler):
    def __init__(self):
        # 用于记录正在处理的病人文件夹，防止同一次写入触发多次处理
        self.processing_folders = set()
        self.lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return

        dir_path = os.path.dirname(event.src_path)
        # 只有 report_tmp 内的变化才触发
        if os.path.basename(dir_path) == "report_tmp":
            patient_folder = os.path.dirname(dir_path)
            self.check_and_process(patient_folder)

    def check_and_process(self, patient_folder):
        if not os.path.isdir(patient_folder):
            return

        with self.lock:
            # 如果这个文件夹正在被处理，就直接跳过（防抖）
            if patient_folder in self.processing_folders:
                return
            # 标记为正在处理
            self.processing_folders.add(patient_folder)

        # 等待文件完全写入磁盘
        time.sleep(DEBOUNCE_TIME)

        try:
            print(f"🚀 监听到新结节数据，开始增量处理: {patient_folder}")
            self.process_patient_data(patient_folder)
        finally:
            # 处理完成后，移除标记，允许下次新文件再次触发
            with self.lock:
                self.processing_folders.discard(patient_folder)

    def process_patient_data(self, patient_folder_path):
        report_tmp = os.path.join(patient_folder_path, "report_tmp")
        # 目标文件夹：SonoMind/patient/{patient_id}
        target_dir = os.path.join(OUTPUT_ROOT, PATIENT_ID)

        os.makedirs(target_dir, exist_ok=True)
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        # ================= 步骤1：仅复制以 _original 结尾的图片 =================
        try:
            copied_count = 0
            for filename in os.listdir(report_tmp):
                # 严格筛选：文件名（不含扩展名）以 _original 结尾
                name_without_ext, ext = os.path.splitext(filename)
                if (ext.lower() in image_extensions and 
                    name_without_ext.endswith("_original")):
                    src_img = os.path.join(report_tmp, filename)
                    dst_img = os.path.join(target_dir, filename)
                    shutil.copy2(src_img, dst_img)
                    copied_count += 1
                    print(f"  📸 复制原始图: {filename}")
            if copied_count == 0:
                print("  ℹ️ 本次未找到符合条件的 _original 图片")
        except Exception as e:
            print(f"  ❌ 复制图片时发生错误: {e}")

        # ================= 步骤2：解析 txt 并生成/合并 JSON =================
        new_nodule_dict = {}
        txt_files = [f for f in os.listdir(report_tmp) if f.endswith(".txt")]
        txt_files.sort()

        for txt_file in txt_files:
            txt_path = os.path.join(report_tmp, txt_file)
            nodule_id_str = os.path.splitext(txt_file)[0]

            try:
                nodule_id_int = int(nodule_id_str)
            except ValueError:
                continue

            location = "Unknown"
            size_mm = 0.0

            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        location = lines[0].strip()
                    if len(lines) > 1:
                        sizes = re.findall(r"([\d.]+)cm", lines[1])
                        if sizes:
                            max_size_cm = max([float(x) for x in sizes])
                            size_mm = round(max_size_cm * 10, 1)
            except Exception as e:
                print(f"  ⚠️ 解析 {txt_file} 失败: {e}")
                continue

            # 匹配主图（仅匹配 _original 结尾的图片）
            main_img_name = ""
            target_files = os.listdir(target_dir)
            
            # 优先匹配结节ID_original.png 格式
            expected_original = f"{nodule_id_str}_original.png"
            if expected_original in target_files:
                main_img_name = expected_original
            else:
                # 遍历所有 _original 结尾的图片，匹配结节ID开头
                for f in target_files:
                    name_without_ext, ext = os.path.splitext(f)
                    if (name_without_ext.endswith("_original") and 
                        name_without_ext.startswith(f"{nodule_id_str}_") and
                        ext.lower() in image_extensions):
                        main_img_name = f
                        break

            # 将新解析的数据存入字典
            new_nodule_dict[nodule_id_int] = {
                "nodule_id": nodule_id_int,
                "location": location,
                "size_mm": size_mm,
                "img_dir": main_img_name
            }

        # ================= 步骤3：读取旧 JSON 并合并写入 =================
        if new_nodule_dict:
            # JSON 文件名与 patient_id 保持一致
            json_path = os.path.join(target_dir, f"{PATIENT_ID}.json")
            existing_data = []

            # 如果旧的 JSON 已经存在，先读取
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ 读取旧 JSON 失败，将创建新文件: {e}")

            # 按 nodule_id 合并新旧数据
            final_nodule_dict = {item['nodule_id']: item for item in existing_data}
            final_nodule_dict.update(new_nodule_dict)

            # 按 nodule_id 排序后写入
            final_nodule_list = sorted(final_nodule_dict.values(), key=lambda x: x['nodule_id'])

            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_nodule_list, f, indent=4, ensure_ascii=False)
                print(f"  ✅ 成功更新 JSON (当前共 {len(final_nodule_list)} 个结节): {json_path}")
            except Exception as e:
                print(f"  ❌ 写入 JSON 失败: {e}")


if __name__ == "__main__":
    # 启动前验证所有配置
    errors = []
    if not os.path.exists(WATCH_DIRECTORY):
        errors.append(f"监控目录不存在: {WATCH_DIRECTORY} (来自 SonoPilot/RobotServer/config.py)")
    if not os.path.exists(OUTPUT_ROOT):
        print(f"ℹ️ 输出根目录不存在，将自动创建: {OUTPUT_ROOT}")
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if not PATIENT_ID:
        errors.append("病人ID为空 (来自 main_UI.py 的 patient_id 变量)")

    if errors:
        print("❌ 配置验证失败:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # 打印启动信息
    print("="*50)
    print("🤖 Sonopilot2Sonomind 数据同步服务")
    print("="*50)
    print(f"📡 监控目录: {WATCH_DIRECTORY}")
    print(f"📤 输出目录: {os.path.abspath(os.path.join(OUTPUT_ROOT, PATIENT_ID))}")
    print(f"🆔 病人ID: {PATIENT_ID}")
    print(f"⏱️ 防抖时间: {DEBOUNCE_TIME}秒")
    print("="*50)
    print("✅ 服务已启动，按 Ctrl+C 停止")
    print("="*50)

    # 启动文件监控
    event_handler = NoduleHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 监控服务已停止")
    observer.join()
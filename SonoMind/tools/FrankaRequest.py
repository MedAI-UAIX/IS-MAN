
import requests
from typing import Any, Dict, Optional
import threading
import queue
from flask import Flask, request, jsonify
from werkzeug.serving import make_server
import atexit
import subprocess
import sys
import threading
import queue
from flask import Flask, request, jsonify
from werkzeug.serving import make_server
import time

def kill_port(port=5006):
    """运行前强制释放端口"""
    try:
        print(f'try to kill port: {port}')
        result = subprocess.run(
            ['fuser', '-k', f'{port}/tcp'],
            capture_output=True,
            text=True
        )
        time.sleep(2)
        if result.returncode == 0:
            print(f'[INFO] 已释放端口 {port}')
        else:
            # fuser 没发现进程时也会返回非0，属于正常情况
            pass
    except FileNotFoundError:
        print(f'[WARN] 未找到 fuser 命令，请手动确保端口 {port} 未被占用')
    except Exception as e:
        print(f'[WARN] 清理端口失败: {e}')


class FrankaClient:
    def __init__(self, base_url: str = "127.0.0.1", port=5004, timeout: float = 5.0):
        self.base_url = 'http://{}:{}'.format(base_url, port)
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params, timeout=self.timeout)
        # print(r)
         

        try:
            r.raise_for_status()
            return r.json()
        except:
            return {'message': 'error'}

    def get_6_keypoint(self):
        print("get keypoint")
        return self._get("/franka/get_6_keypoint")  
    def get_abdomen_keypoint(self):
        print("get abdomen keypoint")
        return self._get("/franka/get_abdomen_keypoint")
    def force_calibration(self):
        print("force calibration")
        return self._get("/franka/force_calibration")
    def force_up(self):
        print("force up")
        return self._get("/franka/force_up")
    def force_down(self):
        print("force up")
        return self._get("/franka/force_down")
    
    def pause(self):
        print("pause")
        return self._get("/franka/pause")
    def speed_up(self):
        print("speed up")
        return self._get("/franka/speed_up")
    def speed_down(self):
        print("speed up")
        return self._get("/franka/speed_down")
    
    def go_home(self):
        print("go home")
        return self._get("/franka/go_home")
    def scanning_left(self):
        print("scanning left")
        return self._get("/franka/scanning_left")
    def scanning_left_longitudinal(self):
        print("scanning left longitudinal")
        return self._get("/franka/scanning_left_longitudinal")
    def left_2_right(self):
        print("from left to right")
        return self._get("/franka/left_2_right")
    def scanning_right(self):
        print("scanning right")
        return self._get("/franka/scanning_right")
    def scanning_right_longitudinal(self):
        print("scanning right longitudinal")
        return self._get("/franka/scanning_right_longitudinal")
    def scanning_isthmus(self):
        print("scanning isthmus")
        return self._get("/franka/scanning_isthmus")  
    def scanning_liver(self):
        print("scanning liver")
        return self._get("/franka/scanning_liver")
    def start_recording(self):
        print("start recording")
        return self._get("/franka/start_recording")
    def stop_recording(self):
        print("stop recording")
        return self._get("/franka/stop_recording")
    def start_for_scanning(self):
        print("start for scanning")
        return self._get("/franka/start_for_scanning")
    def stop_for_scanning(self):
        print("stop for scanning")
        return self._get("/franka/stop_for_scanning")
    def impedance_control(self):
        print("Switching to impedance control")
        return self._get("/franka/impedance_control")
    def hybrid_controller(self):
        print("Switching to hybrid controller")
        return self._get("/franka/hybrid_controller")
    def cartesian_control(self):
        print("Switching to cartesian control")
        return self._get("/franka/cartesian_control")
    def admittance_control(self):
        print("Switching to admittance_controll")
        return self._get("/franka/admittance_control")

    def ControlRobotArm(self, action):
        if action == 'start':
            # self.go_home()
            self.start_for_scanning()
        elif action == 'end':
            self.stop_for_scanning()
        
    def ControlForce(self, action):
        if action == "calibrate":
            self.force_calibration()
        elif action == "up":
            self.force_up()
        elif action == "down":
            self.force_down()
        
    def ControlMotion(self, action):
        if action == "pause":
            # self.stop_recording()
            # self.stop_for_scanning
            self.pause()
        elif action == "speed_up":
            self.speed_up()
        elif action == "speed_down":
            self.speed_down()
        # elif action == "go_home":
        #     self.go_home()
        # elif action == "scanning_left":
        #     self.scanning_left()
        # elif action == "scanning_left_longitudinal":
        #     self.scanning_left_longitudinal()
        # elif action == "left_2_right":
        #     self.left_2_right()
        # elif action == "scanning_right":
        #     self.scanning_right()
        # elif action == "scanning_right_longitudinal":
        #     self.scanning_right_longitudinal()
        # elif action == "scanning_isthmus":
        #     self.scanning_isthmus()
        # elif action == "stop_scanning":
        #     self.stop_for_scanning()
        
        
    def SwitchControlMode(self, mode):
        if mode == "impedance":
            self.impedance_control()
        elif mode == "hybrid":
            self.hybrid_controller()
        elif mode == "cartesian":
            self.cartesian_control()
        elif mode == "admittance":
            self.admittance_control()
        
    def DetectKeypoints(self, region):
        if region == "neck":
            self.get_6_keypoint()
        elif region == "abdomen":
            self.get_abdomen_keypoint()
        
    def UltrasoundScan(self, organ, region, direction):
        organ = organ.lower()
        region = region.lower()

        if organ == "thyroid" :
            if region == "left":
                direction = direction.lower()
                if direction == "longitudinal":
                    self.scanning_left_longitudinal()
                elif direction == "transverse":
                    self.scanning_left()
            elif region == "right":
                direction = direction.lower()
                if direction == "longitudinal":
                    self.scanning_right_longitudinal()
                elif direction == "transverse":
                    self.scanning_right()
            elif region == "isthmus":
                self.scanning_isthmus()
        elif organ == "carotid":
            if region == "left":
                self.scanning_left_longitudinal()
            elif region == "right":
                self.scanning_right_longitudinal()
        elif organ == "liver":
            self.scanning_liver()
        else:
            print("Invalid organ")


class FlaskServerState(threading.Thread):
    def __init__(self, host='0.0.0.0', port=5006):
        super().__init__(daemon=True)
        kill_port(port)
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        self.state_queue = queue.Queue()
        self.message_queue = queue.Queue()
        self.node_queue = queue.Queue()

        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule('/agent/recieve_info_state',
                              view_func=self.receive_info_state,
                              methods=['POST'])
        self.app.add_url_rule('/agent/recieve_info_message',
                              view_func=self.receive_info_message,
                              methods=['POST'])
        self.app.add_url_rule('/agent/recieve_info_node',
                              view_func=self.receive_info_node,
                              methods=['POST'])

    def _extract_message(self):
        data = request.get_json(silent=True)
        if data is None:
            return None, '请求不是合法 JSON'
        if 'message' not in data:
            return None, 'JSON 中缺少 message 字段'
        return data['message'], None

    def receive_info_state(self):
        message, err = self._extract_message()
        if err:
            return jsonify({'code': 400, 'msg': err}), 400

        self.state_queue.put(message)
        print(f'[STATE] {message}')
        return jsonify({'code': 200, 'msg': 'ok', 'data': message})

    def receive_info_message(self):
        message, err = self._extract_message()
        if err:
            return jsonify({'code': 400, 'msg': err}), 400

        self.message_queue.put(message)
        print(f'[MESSAGE] {message}')
        return jsonify({'code': 200, 'msg': 'ok', 'data': message})

    def receive_info_node(self):
        message, err = self._extract_message()
        if err:
            return jsonify({'code': 400, 'msg': err}), 400

        self.node_queue.put(message)
        print(f'[NODE] {message}')
        return jsonify({'code': 200, 'msg': 'ok', 'data': message})

    def wait_for_state(self, timeout=None):
        """
        没有状态时一直阻塞；
        timeout=None 表示永久等
        """
        try:
            return self.state_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for_message(self, timeout=None):
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for_node(self, timeout=None):
        try:
            return self.node_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def run(self):
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True
        )

# class FlaskServerState(threading.Thread):
#     def __init__(self, host='0.0.0.0', port=5004):
#         # 注意：先不设 daemon，或确保主线程主动关闭
#         super().__init__(daemon=True)
#         self.host = host
#         self.port = port
#         self.app = Flask(__name__)
#         self._server = None  # 保存 server 实例，用于 shutdown

#         self.state_queue = queue.Queue()
#         self.message_queue = queue.Queue()
#         self.node_queue = queue.Queue()

#         self._register_routes()

#     def _register_routes(self):
#         self.app.add_url_rule('/agent/recieve_info_state',
#                               view_func=self.receive_info_state,
#                               methods=['POST'])
#         self.app.add_url_rule('/agent/recieve_info_message',
#                               view_func=self.receive_info_message,
#                               methods=['POST'])
#         self.app.add_url_rule('/agent/recieve_info_node',
#                               view_func=self.receive_info_node,
#                               methods=['POST'])

#     def _extract_message(self):
#         data = request.get_json(silent=True)
#         if data is None:
#             return None, '请求不是合法 JSON'
#         if 'message' not in data:
#             return None, 'JSON 中缺少 message 字段'
#         return data['message'], None

#     def receive_info_state(self):
#         message, err = self._extract_message()
#         if err:
#             return jsonify({'code': 400, 'msg': err}), 400
#         self.state_queue.put(message)
#         print(f'[STATE] {message}')
#         return jsonify({'code': 200, 'msg': 'ok', 'data': message})

#     def receive_info_message(self):
#         message, err = self._extract_message()
#         if err:
#             return jsonify({'code': 400, 'msg': err}), 400
#         self.message_queue.put(message)
#         print(f'[MESSAGE] {message}')
#         return jsonify({'code': 200, 'msg': 'ok', 'data': message})

#     def receive_info_node(self):
#         message, err = self._extract_message()
#         if err:
#             return jsonify({'code': 400, 'msg': err}), 400
#         self.node_queue.put(message)
#         print(f'[NODE] {message}')
#         return jsonify({'code': 200, 'msg': 'ok', 'data': message})

#     def wait_for_state(self, timeout=None):
#         try:
#             return self.state_queue.get(timeout=timeout)
#         except queue.Empty:
#             return None

#     def wait_for_message(self, timeout=None):
#         try:
#             return self.message_queue.get(timeout=timeout)
#         except queue.Empty:
#             return None

#     def wait_for_node(self, timeout=None):
#         try:
#             return self.node_queue.get(timeout=timeout)
#         except queue.Empty:
#             return None

#     def run(self):
#         # 用 make_server 替代 app.run，支持 shutdown
#         self._server = make_server(
#             self.host, self.port, self.app,
#             threaded=True,
#             # Werkzeug 默认已开启 SO_REUSEADDR，通常无需额外设置
#         )
#         self._server.serve_forever()

#     def shutdown(self):
#         """优雅关闭，释放端口"""
#         if self._server:
#             self._server.shutdown()


    
if __name__ == "__main__":
    franka = FrankaClient(port=5004)
    state_server = FlaskServerState(port=5006)
    state_server.start()
    
    # s = franka.ControlRobotArm(action='start')
    # print('cmd状态:', s)

    s = franka.ControlRobotArm(action='end')
    print('cmd状态:', s)


    # s = franka.ControlForce(action='calibrate')
    # print('cmd状态:', s)

    # s = franka.ControlForce(action='up')
    # print('cmd状态:', s)

    # s = franka.ControlForce(action='down')
    # print('cmd状态:', s)

    
    # s = franka.ControlMotion(action='speed_up')  # pause    speed_up   speed_down
    # print('cmd状态:', s)

    
    # s = franka.SwitchControlMode(mode='impedance')  # impedance   hybrid  cartesian  admittance
    # print('cmd状态:', s)

    # s = franka.DetectKeypoints(region='neck')  # impedance   hybrid  cartesian  admittance
    # print('cmd状态:', s)

    # s = franka.UltrasoundScan(organ='thyroid', region='left', direction='transverse')  # impedance   hybrid  cartesian  admittance
    # print('cmd状态:', s)

    state = state_server.wait_for_state()   # 没有就一直阻塞
    print('robot状态:', state)
    # franka.SwitchControlMode("impedance")
    # franka.start_for_scanning()
    # franka.stop_for_scanning()
    # franka.impedance_control()
import subprocess
import time
import signal

def start_command():
    # 定义要执行的指令
    command_franka = "/home/usai/auto_RUSS/real_implement/tools/RobotServer/run_franka.sh"
    command_seg = "/home/usai/auto_RUSS/real_implement/tools/RobotServer/run_seg.sh"
    command_cam = '/home/usai/auto_RUSS/real_implement/tools/RobotServer/run_cam.sh'
    command_publish_us = '/home/usai/auto_RUSS/real_implement/tools/RobotServer/run_publish_us.sh'
    # command_contact= '/home/usai/auto_RUSS/real_implement/tools/RobotServer/run_contact.sh'
    command_recorder = '/home/usai/auto_RUSS/real_implement/tools/RobotServer/run_recoder.sh'
    # command_agent = '/home/usai/auto_RUSS/R_UI/run_agent_server.sh'
    try:
        proc = subprocess.Popen(['gnome-terminal', '--', command_franka])
        time.sleep(2)
        subprocess.Popen(['gnome-terminal', '--', command_cam])
        subprocess.Popen(['gnome-terminal', '--', command_publish_us])
        subprocess.Popen(['gnome-terminal', '--', command_seg])
        # subprocess.Popen(['gnome-terminal', '--', command_contact])
        subprocess.Popen(['gnome-terminal', '--', command_recorder])
        # subprocess.Popen(['gnome-terminal', '--', command_agent])
     
        print('sucess')
        return True
    except:
        print('error')
        return False



if __name__ == '__mian__':
    ret = start_command()

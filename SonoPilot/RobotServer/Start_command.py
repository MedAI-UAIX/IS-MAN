import subprocess
import time
import signal

def start_command():
    # 定义要执行的指令
    command_franka = "/home/usai/IS-MAN/SonoPilot/RobotServer/run_franka.sh"
    command_seg = "/home/usai/IS-MAN/SonoPilot/RobotServer/run_seg.sh"
    command_publish_us = '/home/usai/IS-MAN/SonoPilot/RobotServer/run_publish_us.sh'
    command_recorder = '/home/usai/IS-MAN/SonoPilot/RobotServer/run_recoder.sh'
    command_sonopilot2sonomind = '/home/usai/IS-MAN/SonoPilot/RobotServer/run_sonopilot2sonomind.sh'

    try:
        proc = subprocess.Popen(['gnome-terminal', '--', command_franka])
        time.sleep(2)
        subprocess.Popen(['gnome-terminal', '--', command_publish_us])
        subprocess.Popen(['gnome-terminal', '--', command_seg])
        subprocess.Popen(['gnome-terminal', '--', command_recorder])
        subprocess.Popen(['gnome-terminal', '--', command_sonopilot2sonomind])
     
        print('sucess')
        return True
    except:
        print('error')
        return False

if __name__ == '__mian__':
    ret = start_command()

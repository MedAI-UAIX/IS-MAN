import subprocess
import time
import os

def start_command():
    # Get the directory where the current script is located (RobotServer directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define scripts with relative paths
    command_franka = os.path.join(script_dir, 'run_franka.sh')
    command_seg = os.path.join(script_dir, 'run_seg.sh')
    command_cam = os.path.join(script_dir, 'run_cam.sh')
    command_publish_us = os.path.join(script_dir, 'run_publish_us.sh')
    command_recorder = os.path.join(script_dir, 'run_recoder.sh')  # 注意文件名是recoder
    
    # Add execute permission to the script
    for cmd in [command_franka, command_seg, command_cam, command_publish_us, command_recorder]:
        os.chmod(cmd, 0o755)
    
    try:
        # Prefer using gnome-terminal; if it fails, try xterm or run directly in the background
        terminals = ['gnome-terminal', 'xterm', 'konsole']
        terminal = next((t for t in terminals if subprocess.call(['which', t], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0), None)
        
        if terminal:
            def run_in_terminal(cmd):
                subprocess.Popen([terminal, '--', cmd])
        else:
            def run_in_terminal(cmd):
                subprocess.Popen(['bash', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        run_in_terminal(command_franka)
        time.sleep(2)
        run_in_terminal(command_cam)
        run_in_terminal(command_publish_us)
        run_in_terminal(command_seg)
        run_in_terminal(command_recorder)
        
        print('success')
        return True
    except Exception as e:
        print(f'error: {e}')
        return False

if __name__ == '__main__':
    ret = start_command()
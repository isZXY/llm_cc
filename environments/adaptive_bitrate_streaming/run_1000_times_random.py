import os
import subprocess

# 基础路径和文件夹设置
_base_dir = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/'
artifacts_base_dir = _base_dir + 'artifacts_random_trace/'

# 创建一个新的文件夹，确保每次仿真时有一个唯一的文件夹
def run_simulation(simulation_num):
    # 为每次仿真创建唯一的文件夹
    simulation_folder = os.path.join(artifacts_base_dir, str(simulation_num))
    
    # 如果文件夹不存在，则创建它
    if not os.path.exists(simulation_folder):
        os.makedirs(simulation_folder)
    
    # 运行仿真，假设仿真是通过调用一个外部脚本运行的（如运行 Python 脚本或命令行程序）
    # 这里我们假设 `run_abr_generate_random_trace.py` 是你用来运行仿真的脚本
    # 你可以根据实际情况修改为相应的调用方式
    command = [
        'python', _base_dir + 'run_abr_generate_random_trace.py',  # 仿真脚本
        '--output_dir', simulation_folder  # 指定输出路径
    ]
    
    # 使用 subprocess 运行仿真脚本
    subprocess.run(command)

def main():
    # 运行1000次仿真
    for i in range(1, 1001):
        run_simulation(i)
        print(f"Simulation {i} completed and saved to {artifacts_base_dir}{i}")

if __name__ == "__main__":
    main()

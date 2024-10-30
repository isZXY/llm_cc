import os
import pandas as pd

HEADER = ['(double)time_on_trace/1e3','max_tmp','(double)sage_info.rtt/100000.0','(double)sage_info.rttvar/1000.0','(double)sage_info.rto/100000.0','(double)sage_info.ato/100000.0','(double)sage_info.pacing_rate/125000.0/BW_NORM_FACTOR','(double)sage_info.delivery_rate/125000.0/BW_NORM_FACTOR','sage_info.snd_ssthresh','sage_info.ca_state','rtt_s.get_avg()','rtt_s.get_min()','rtt_s.get_max()','rtt_m.get_avg()','rtt_m.get_min()','rtt_m.get_max()','rtt_l.get_avg()','rtt_l.get_min()','rtt_l.get_max()','thr_s.get_avg()','thr_s.get_min()','thr_s.get_max()','thr_m.get_avg()','thr_m.get_min()','thr_m.get_max()','thr_l.get_avg()','thr_l.get_min()','thr_l.get_max()','rtt_rate_s.get_avg()','rtt_rate_s.get_min()','rtt_rate_s.get_max()','rtt_rate_m.get_avg()','rtt_rate_m.get_min()','rtt_rate_m.get_max()','rtt_rate_l.get_avg()','rtt_rate_l.get_min()','rtt_rate_l.get_max()','rtt_var_s.get_avg()','rtt_var_s.get_min()','rtt_var_s.get_max()','rtt_var_m.get_avg()','rtt_var_m.get_min()','rtt_var_m.get_max()','rtt_var_l.get_avg()','rtt_var_l.get_min()','rtt_var_l.get_max()','inflight_s.get_avg()','inflight_s.get_min()','inflight_s.get_max()','inflight_m.get_avg()','inflight_m.get_min()','inflight_m.get_max()','inflight_l.get_avg()','inflight_l.get_min()','inflight_l.get_max()','lost_s.get_avg()','lost_s.get_min()','lost_s.get_max()','lost_m.get_avg()','lost_m.get_min()','lost_m.get_max()','lost_l.get_avg()','lost_l.get_min()','lost_l.get_max()','dr_w_mbps-l_w_mbps','time_delta','rtt_rate','l_w_mbps/BW_NORM_FACTOR','acked_rate','(pre_dr_w_mbps>0.0)?(dr_w_mbps/pre_dr_w_mbps):dr_w_mbps','(double)(dr_w_max*sage_info.min_rtt)/(cwnd_bits)','(dr_w_mbps)/BW_NORM_FACTOR','cwnd_unacked_rate','(pre_dr_w_max>0.0)?(dr_w_max/pre_dr_w_max):dr_w_max','dr_w_max/BW_NORM_FACTOR','reward','(cwnd_rate>0.0)?round(log2f(cwnd_rate)*1000)/1000.:log2f(0.0001))']

if __name__ == '__main__':

    # 要遍历的路径
    directory = "/data3/wuduo/xuanyu/llmcc/datasets/CC/txt"
    store_dir = "/data3/wuduo/xuanyu/llmcc/datasets/CC/csv"
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            logfile_name = os.path.join(root, file)
            file_name_without_extension = os.path.splitext(file)[0]
            csv_file_name = file_name_without_extension + ".csv"
            store_path = os.path.join(store_dir, csv_file_name)
            # 判断文件是否为空
            if os.path.getsize(logfile_name) == 0:
                continue  # 如果文件为空，跳过

            df = pd.DataFrame(columns=HEADER)

            with open(logfile_name, 'r') as file:
                for line in file:
                    row = line.strip().split()

                    try:
                        # 将新数据行转为 DataFrame，并检查列数是否匹配
                        if len(row) == len(df.columns):
                            new_row = pd.DataFrame([row], columns=df.columns)
                            # 使用 pd.concat 追加新行到现有的 DataFrame
                            df = pd.concat([df, new_row], ignore_index=True)
                        else:
                            raise ValueError(f"列数不匹配：文件 {logfile_name} 中的行 {row} 的列数是 {len(row)}，而 DataFrame 的列数是 {len(df.columns)}")
                    except Exception as e:
                    # 捕获并打印异常
                        print(f"处理文件 {logfile_name} 时出错: {e}")
                        continue  # 出现异常后，继续处理下一个文件或行

                    df.to_csv(store_path, index=False, header=True)
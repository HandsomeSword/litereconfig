'''
LiteReconfig w/ the cost benefit analyzer, a.k.a. the full version.

Example usage:
python LiteReconfig.py --gl 0 --lat_req 33.3 --mobile_device=tx2
'''

import argparse
import numpy as np
from tqdm import tqdm
from helper_online_dev import SchedulerCBOnline
from helper_online import MBODF

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate SmartAdapt_CB.')
parser.add_argument('--gl', dest='gl', type=int, required=True, help='GPU contention level.')
parser.add_argument('--lat_req', dest='lat_req', type=float, help='Latency requirement in msec.')
parser.add_argument('--mobile_device', dest='mobile_device', required=True, help='tx2 or xv.')
parser.add_argument('--cost_filename', dest='cost_filename',
  default='models/SmartAdapt_cost_20211009.pb', help='Cost weight file.')
parser.add_argument('--benefit_filename', dest='benefit_filename',
  default='models/SmartAdapt_benefit_20211009.pb', help='Benefit weight file.')
parser.add_argument('--quick', dest='quick', type=int, help='Whether to run on 10% dataset.')
parser.add_argument('--output', dest='output', default='test/executor_LiteReconfig.txt',
  help='Output filename.')
parser.add_argument('--tv_version', dest='tv_version', help='torchvision version, e.g., 0.5.0')
parser.add_argument('--dataset_prefix', dest='dataset_prefix', help='Path to ILSVRC2015 dir.')
args = parser.parse_args()



#在这个txt文件中存放的是视频的路径和帧数
metadata = "test/VID_testvideo_V2.txt"
with open(metadata) as fin:
    lines = fin.readlines()
if args.quick:  # 10% test dataset
    lines = lines[::10]





#video_dirs是视频的路径，frame_cnts是视频的帧数
video_dirs = [x.strip().split()[0] for x in lines]
frame_cnts = {line.split()[0]:int(line.split()[1]) for line in lines}






#contention_levels是用户设置的各个硬件的优先级，这里只设置了gpu的优先级
contention_levels = {'cpu_level': 0, 'mem_bw_level': 0, 'gpu_level': args.gl}







#创建了一个scheduler，这个scheduler是用来调度的，调度的时候会用到用户设置的优先级，延迟要求，成本权重，收益权重，是否是移动设备，
#torchvision版本，数据集路径前缀。成本权重和收益权重暂时不知道什么意思 
# TODO
scheduler = SchedulerCBOnline(contention_levels=contention_levels, user_requirement=args.lat_req,
                              cost_filename=args.cost_filename,
                              benefit_filename=args.benefit_filename,
                              p95_requirement=True, mobile_device=args.mobile_device,
                              tv_version=args.tv_version, dataset_prefix=args.dataset_prefix)






#创建了两个输出文件的名字。一个是检测的输出文件，一个是延迟的输出文件
filename_pre = args.output.rsplit(".", 1)[0]
filename_det = f"{filename_pre}_g{args.gl}_lat{int(args.lat_req)}_{args.mobile_device}_det.txt"
filename_lat = f"{filename_pre}_g{args.gl}_lat{int(args.lat_req)}_{args.mobile_device}_lat.txt"
with open(filename_det, "w") as fout_det, open(filename_lat, "w") as fout_lat:
    






    #创建了一个执行器，这个执行器是用来执行的，执行的时候会用到检测的输出文件和延迟的输出文件
    executor = MBODF(feat="RPN", kernel="FRCNN+", frcnn_weight="models/ApproxDet.pb",
                     fout_det=fout_det, fout_lat=fout_lat)
    tqdm_desc = f"LiteReconfig, g{args.gl}/{args.lat_req:.1f} ms lat_req, on {len(video_dirs)} videos"







    #这里的tqdm就是整了个进度条，对数据循环没有任何影响，每次循环就是取出视频的编号和路径。desc这个参数是用来显示进度条名字的
    for video_idx, video_dir in tqdm(enumerate(video_dirs), desc=tqdm_desc):
        frame_cnt, frame_idx = frame_cnts[video_dir], 0






        #feature_cache暂时不理解 
        # TODO
        # nobj应该是物体的数量，objsize应该是物体的大小
        feature_cache = {"nobj": 0, "objsize": 0, "RPN": np.zeros((1024,)), "CPoP": np.zeros((31,))}






        #遍历视频的每一个帧，frame_idx是帧的编号，frame_cnt是视频的帧数
        while frame_idx < frame_cnt:
            





            #调用这个调度器，这个调度器需要视频路径，视频编号，帧编号，特征缓存，优先级，返回的内容暂时不理解
            # TODO 
            # config里的si是GOF的大小。
            # 
            config, img_pil, run_log = scheduler.schedule(video_dir, video_idx, frame_idx,
                                                          feature_cache, contention_levels)
            si, shape, nprop, tracker_name, ds = config








            # 这里的GOF结合论文内容就是每次处理的一小段帧数。si就是GOF的大小。从这里可以看出，这个si并不是一个固定的值，他是通过scheduler之后，得到的一个数值
            # 也就是说，会分析每段帧开头的第一个帧，然后通过某种方法确定这个si的值。
            # 阅读后面的代码可知，这个si也并不是通过复杂的计算得到的，而是预设一些值，根据某种规则选择一个值。
            frame_cnt_GoF = min(si, frame_cnt-frame_idx)







            #执行器执行，需要config，GOF，视频路径，帧编号，图片，特征缓存，运行日志
            # TODO
            executor.run(config, frame_cnt_GoF, video_dir, frame_idx, img_pil, feature_cache, run_log)









            # 处理完成之后，去下一段帧
            frame_idx += frame_cnt_GoF

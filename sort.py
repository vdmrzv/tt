import os
import sys
import cv2
import pdb
import pickle
import datetime
import numpy as np
import pandas as pd

from functools import partial
from multiprocessing.pool import ThreadPool
from requests.exceptions import ConnectionError

mega_log_dict = {}

bad_dirs = ['/data/pictures/2020/07/09/test234',
            '/data/pictures/2020/08/28/agrotest-4217',
            '/data/pictures/2020/08/28/undefined',
            '/data/pictures/2020/08/31/testing',
            '/data/pictures/2020/09/01/agrotest-4217',
            '/data/pictures/2020/09/03/undefined',
            '/data/pictures/2020/09/08/undefined',
            '/data/pictures/2020/09/09/undefined',
            '/data/pictures/2020/09/10/undefined',
            '/data/pictures/2020/10/27/agrodroid-20050025',
            '/data/pictures/2020/11/03/agrodroid-20050025']

def run_sorter(num_threads, img_folders):
    pool = ThreadPool(num_threads)
    results = pool.imap_unordered(img_sorter, img_folders)

    for r in results:
        print(r)

    pool.close()
    pool.join()

def img_sorter(img_folder):

    imgs = sorted(os.listdir(img_folder))
    droid_log_list = []
    droid_log_dict = {}

    if len(imgs) > 5:
        # Parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners = 100,
                              qualityLevel = 0.3,
                              minDistance = 7,
                              blockSize = 7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = (15,15),
                         maxLevel = 2,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        for prev, cur in zip(imgs, imgs[1:]):

            old_frame_path = os.path.join(img_folder, prev)
            new_frame_path = os.path.join(img_folder, cur)

            # Take first frame and find corners in it
            old_frame = cv2.imread(old_frame_path)
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)
            new_frame = cv2.imread(new_frame_path)
            new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            if isinstance(p0, (np.ndarray)):

                # Calculate optical flow using various algos
                try:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
                except:
                    print(img_folder, prev ,cur)

                flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]

                # ------------------------ METRICS ------------------------------
                ecc_color = cv2.computeECC(old_frame, new_frame)
                sum_diff_color = abs(int(new_frame.sum()) - int(old_frame.sum()))
                err_sum = err.sum()
                mag_sum = mag.sum()
                len_p1 = len(p1)

                metrics = [err_sum > 1500,
                           ecc_color < 1.0,
                           mag_sum > 500000,
                           sum_diff_color > 1000000,
                           len_p1 < 50]
                score = sum(metrics)
                moving = False

                if score >= 3:
                    moving = True

                # Update previous points
                p0 = good_new.reshape(-1,1,2)

                droid_log_list.append({'moving':moving,
                                'score':score,
                                'prev':os.path.join(img_folder, prev),
                                'cur':os.path.join(img_folder, cur),
                                'err_sum':err_sum,
                                'ecc_color':ecc_color,
                                'mag_sum':mag_sum,
                                'sum_diff_color':sum_diff_color,
                                'len_p1':len_p1,
                                'seq_num':None})
            else:
                print(type(p0), img_folder, prev, cur)
                moving = False

            # Update the previous frame
            old_gray = new_gray.copy()

        seq_counter = 0
        for prev, cur in zip(droid_log_list, droid_log_list[1:]):
            prev['seq_num'] = seq_counter
            if (abs(cur['ecc_color'] - prev['ecc_color'])/cur['ecc_color']*100 > 20 and
                cur['sum_diff_color'] > 9000000):
                seq_counter += 1

        cur['seq_num'] = seq_counter

        df = pd.DataFrame(droid_log_list)

        for i in range(len(df.seq_num.unique())):
            seq_df = df.loc[df['seq_num'] == i]

            if len(seq_df) <= 5:
                df.loc[df['seq_num'] == i] = df.loc[df['seq_num'] == i].assign(moving='False')
            else:
                if seq_df['score'].mean() > 2.8:
                    # make every row moving True
                    df.loc[df['seq_num'] == i] = df.loc[df['seq_num'] == i].assign(moving='True')
                else:
                    # make every row moving False
                    df.loc[df['seq_num'] == i] = df.loc[df['seq_num'] == i].assign(moving='False')

        for i in range(len(df)):
            if i == 0:
                droid_log_dict.update({df.iloc[i]['prev']:df.iloc[i]['moving']})
            else:
                droid_log_dict.update({df.iloc[i]['prev']:df.iloc[i - 1]['moving']})

        mega_log_dict.update(droid_log_dict)
        return f'complete {img_folder}'
    elif len(imgs) <= 5 and len(imgs) > 0:
        for img in imgs:
            img_path = os.path.join(img_folder, img)
            droid_log_dict.update({img_path:'False'})
        mega_log_dict.update(droid_log_dict)
        return f'complete {img_folder}'
    else:
        return f'empty folder {img_folder}'

def find_all_droid_folders(src_dir):
    ''' find all folders of this format /2020/07/01/id_agrodroid-20050029'''
    src_dir_len = len(src_dir.split(os.sep))
    all_droid_folders = []

    for root, dirs, path in os.walk(src_dir):
        if len(root.split(os.sep)) - src_dir_len == 3:
            all_droid_folders.append(root)

    return all_droid_folders

def parse_input():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python3 sort_images.py /path/to/src/folder num_threads')

    src_dir = sys.argv[1]
    num_threads = int(sys.argv[2])

    return src_dir, num_threads

def main():
    exec_time = datetime.datetime.now()
    src_dir, num_threads = parse_input()
    all_droid_folders = sorted(find_all_droid_folders(src_dir))
    print(len(all_droid_folders))
    all_good_folders = [item for item in all_droid_folders if item not in bad_dirs]
    print(len(all_good_folders))
    pdb.set_trace()

    run_sorter(num_threads, all_good_folders)

    with open('sort_dict_' + exec_time.strftime("%d-%m-%Y_%H-%M-%S") + '.pickle', 'wb') as f:
        pickle.dump(mega_log_dict, f)

if __name__ =='__main__':
    main()

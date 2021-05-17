#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@file: image_process.py
@time: 2021/03/29 21:40
@desc: image process lib
"""
# <---------------------------------------------------------------->

# <---------------------------Import Packages---------------------->
import os
import re
import cv2
import shutil
import numpy as np
import pandas as pd
import multiprocessing
from ftplib import FTP
import matplotlib as mpl
import scipy.ndimage as nd
from skimage import measure
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap


# <----------------------------Initilize Reload Classes------------>
class Time():
    """
    time class
    """

    def __init__(self, *args):
        """
        if len *args = 1,means second; if len *args = 3, means hour, minute, second
        :param args: 1 or 3 acceptable args
        """
        if len(args) == 3:
            t = int(args[0]) * 3600 + int(args[1]) * 60 + int(args[2])
            self.h = int(t / 3600)
            self.m = int((t % 3600) / 60)
            self.s = int((t % 3600) % 60)
        elif len(args) == 1:
            sec = int(args[0])
            self.h = int(sec / 3600)
            self.m = int((sec % 3600) / 60)
            self.s = int((sec % 3600) % 60)

    def __str__(self):
        """
        reload print()/str() ...
        :return:
        """
        return ("{:0>2d}:{:0>2d}:{:0>2d}".format(self.h, self.m, self.s))

    def __lt__(self, other):
        """
        reload sorted() and < operator
        :param other:
        :return:
        """
        return (self.h * 3600 + self.m * 60 + self.s) < (other.h * 3600 + other.m * 60 + other.s)

    def __add__(self, other):
        """
        reload + operator
        :param other:
        :return:
        """
        r_h = self.h + other.h
        r_m = self.m + other.m
        r_s = self.s + other.s
        return Time(r_h, r_m, r_s)

    def __sub__(self, other):
        """
        reload - operator
        :param other:
        :return:
        """
        r_h = self.h - other.h
        r_m = self.m - other.m
        r_s = self.s - other.s
        return Time(r_h, r_m, r_s)


def time_norm(times):
    """
    Normlization the time list
    :param times: class Time
    :return: normlizated time list
    """
    t0 = times[0]
    for n, time in enumerate(times):
        times[n] -= t0
    return times


# <----------------------------Initilize plt para------------------>
npblue = ['#000000', '#000002', '#000004', '#000006', '#000008', '#00000A', '#00000C', '#00000E', '#000010',
          '#000012', '#000014', '#000016', '#000018', '#00001A', '#00001C', '#00001E', '#000020', '#000022',
          '#000024', '#000026', '#000028', '#00002A', '#00002C', '#00002E', '#000030', '#000032', '#000034',
          '#000036', '#000038', '#00003A', '#00003C', '#00003E', '#000040', '#000042', '#000044', '#000046',
          '#000048', '#00004A', '#00004C', '#00004E', '#000050', '#000052', '#000054', '#000056', '#000058',
          '#00005A', '#00005C', '#00005E', '#000060', '#000062', '#000064', '#000066', '#000068', '#00006A',
          '#00006C', '#00006E', '#000070', '#000072', '#000074', '#000076', '#000078', '#00007A', '#00007C',
          '#00007E', '#000280', '#000482', '#000684', '#000886', '#000A88', '#000C8A', '#000E8C', '#00108E',
          '#001290', '#001492', '#001694', '#001896', '#001A98', '#001C9A', '#001E9C', '#00209E', '#0022A0',
          '#0024A2', '#0026A4', '#0028A6', '#002AA8', '#002CAA', '#002EAC', '#0030AE', '#0032B0', '#0034B2',
          '#0036B4', '#0038B6', '#003AB8', '#003CBA', '#003EBC', '#0040BE', '#0042C0', '#0044C2', '#0046C4',
          '#0048C6', '#004AC8', '#004CCA', '#004ECC', '#0050CE', '#0052D0', '#0054D2', '#0056D4', '#0058D6',
          '#005AD8', '#005CDA', '#005EDC', '#0060DE', '#0062E0', '#0064E2', '#0066E4', '#0068E6', '#006AE8',
          '#006CEA', '#006EEC', '#0070EE', '#0072F0', '#0074F2', '#0076F4', '#0078F6', '#007AF8', '#007CFA',
          '#007EFC', '#0080FE', '#0282FF', '#0484FF', '#0686FF', '#0888FF', '#0A8AFF', '#0C8CFF', '#0E8EFF',
          '#1090FF', '#1292FF', '#1494FF', '#1696FF', '#1898FF', '#1A9AFF', '#1C9CFF', '#1E9EFF', '#20A0FF',
          '#22A2FF', '#24A4FF', '#26A6FF', '#28A8FF', '#2AAAFF', '#2CACFF', '#2EAEFF', '#30B0FF', '#32B2FF',
          '#34B4FF', '#36B6FF', '#38B8FF', '#3ABAFF', '#3CBCFF', '#3EBEFF', '#40C0FF', '#42C2FF', '#44C4FF',
          '#46C6FF', '#48C8FF', '#4ACAFF', '#4CCCFF', '#4ECEFF', '#50D0FF', '#52D2FF', '#54D4FF', '#56D6FF',
          '#58D8FF', '#5ADAFF', '#5CDCFF', '#5EDEFF', '#60E0FF', '#62E2FF', '#64E4FF', '#66E6FF', '#68E8FF',
          '#6AEAFF', '#6CECFF', '#6EEEFF', '#70F0FF', '#72F2FF', '#74F4FF', '#76F6FF', '#78F8FF', '#7AFAFF',
          '#7CFCFF', '#7EFEFF', '#80FFFF', '#82FFFF', '#84FFFF', '#86FFFF', '#88FFFF', '#8AFFFF', '#8CFFFF',
          '#8EFFFF', '#90FFFF', '#92FFFF', '#94FFFF', '#96FFFF', '#98FFFF', '#9AFFFF', '#9CFFFF', '#9EFFFF',
          '#A0FFFF', '#A2FFFF', '#A4FFFF', '#A6FFFF', '#A8FFFF', '#AAFFFF', '#ACFFFF', '#AEFFFF', '#B0FFFF',
          '#B2FFFF', '#B4FFFF', '#B6FFFF', '#B8FFFF', '#BAFFFF', '#BCFFFF', '#BEFFFF', '#C0FFFF', '#C2FFFF',
          '#C4FFFF', '#C6FFFF', '#C8FFFF', '#CAFFFF', '#CCFFFF', '#CEFFFF', '#D0FFFF', '#D2FFFF', '#D4FFFF',
          '#D6FFFF', '#D8FFFF', '#DAFFFF', '#DCFFFF', '#DEFFFF', '#E0FFFF', '#E2FFFF', '#E4FFFF', '#E6FFFF',
          '#E8FFFF', '#EAFFFF', '#ECFFFF', '#EEFFFF', '#F0FFFF', '#F2FFFF', '#F4FFFF', '#F6FFFF', '#F8FFFF',
          '#FAFFFF', '#FCFFFF', '#FEFFFF', '#FFFFFF']
newcmp = LinearSegmentedColormap.from_list('cmap', npblue)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24, }
dpi = 100
mpl.rcParams['figure.dpi'] = dpi


# <---------------------FTP File Get Method------------------------>
def local_creater(local_path):
    """
    used in ftp_receve function
    :param local_path:
    :return:
    """
    if (os.path.exists(local_path)):
        shutil.rmtree(local_path)
        os.makedirs(local_path)
    else:
        os.makedirs(local_path)
    os.chdir(local_path)
    print(os.getcwd())


def isfile(ftp, file):
    """
    ftp.isfile
    :param ftp:
    :param file:
    :return:
    """
    try:
        ftp.cwd(file)
        ftp.cwd('..')
        return False
    except:
        return True


def ftp_receve(date, local_path):
    """
    receve data from ftp station, decluding subdirs
    :param date:
    :param local_path:
    :return:
    """
    ftp = FTP()
    ftp.connect("10.20.51.123", 888)
    ftp.login("dynamics", "Molecules")
    ftp.set_pasv(False)
    ftp.cwd("Experimental_Data/")
    if date in ftp.nlst():
        ftp.cwd(date)
        local_creater(local_path)
        print("\n+---------- downloading ----------+")
        i = 0
        l = len(ftp.nlst())
        for file in ftp.nlst():
            i += 1
            if isfile(ftp, file):
                file_handle = open(file, "wb").write
                ftp.retrbinary("RETR " + file, file_handle, 1024)
            print("\rDownloading：{:.2%} .".format(i / l), end="")
        print("\n+---------- download over --------+")
    else:
        print("Dir not exist!")
    ftp.quit()


def read_split(local_path):
    """
    split by file name with "_"
    :param local_path:
    :return:
    """
    if (os.path.exists(local_path)):
        print("\n+---------- Spliting ----------+")
        os.chdir(local_path)
        i = 0
        l = len(os.listdir(local_path))
        for file in os.listdir(local_path):
            i += 1
            if "_" in file:
                a = file.split("_")
                if not os.path.exists(a[0]):
                    os.makedirs(local_path + a[0] + "/")
                shutil.move(local_path + file, local_path + a[0] + "/")
            print("\rMoving：{:.2%} .".format(i / l), end="")
        print("\n+---------- Splitover ----------+")
    else:
        print("Dir not exist!!")


def save_png(file):
    """
    save single png
    :param file:
    :return:
    """
    fname = file.split(".")
    f1 = pd.read_csv(file, header=None)
    f1 = f1.dropna()
    a = np.array(f1)
    a = image_mixfilter(a)
    plt.imshow(a, cmap=newcmp, aspect='equal', origin='lower')
    plt.axis('off')
    plt.savefig(fname[0] + ".png", bbox_inches='tight', pad_inches=0)
    return


# <----------------------------Image Process Method---------------->
def image_mixfilter(a):
    """
    filter by gaussian_filyer, medfilter
    :param a: numpy 2-d array as image
    :return:b: numpy 2-d array as filter output
    """
    b = a
    b = signal.medfilt(b, (3, 3))
    # b = nd.gaussian_filter(b, sigma=2)
    return b


def get_hist(a):
    """
    get hist
    :param a: numpy 2-d array as image
    :return: n: 1-d list include data counts; bins: a 1-d list show every data counts refer value
    """
    n, bins = np.histogram(a.ravel(), 256)
    return n, bins


def get_hist_img(b_cv):
    """
    img get hist
    :param b_cv: cv images like CV_8UC3 or CV_8UC1
    :return: n: 1-d or 3-d list include data counts; bins: a 1-d or 3-d list show every data counts refer value
    """
    n, bins, p = plt.hist(b_cv.ravel(), 256)
    return n, bins


def find_ions_mor(a, r1, r=0.5, a1=1, b1=0):
    """
    find ions by morphology
    :param a: numpy 2-d array
    :param r1: threshold area
    :param r: threshold ratio
    :param a1: a corrected bg factor
    :param b1: b corrected bg factor
    :return: contours lens, mean counts minus bg / plt show in outside, contours， b_cut
    """
    bg = a[r1[1]:(r1[1] + r1[3]), (r1[0] + r1[2]):(r1[0] + r1[2] * 2)]
    b = a[r1[1]:(r1[1] + r1[3]), r1[0]:(r1[0] + r1[2])]
    ts = b.max() - (b.max() - b.min()) * r
    b_thresh = cv2.threshold(b, ts, 255, cv2.THRESH_BINARY)[1]  # 阈值分割
    b_thresh = cv2.erode(b_thresh, None, iterations=2)
    b_thresh = cv2.dilate(b_thresh, None, iterations=10)  # 形态学开运算
    b_cut = b_thresh
    contours = measure.find_contours(b_cut, 1)
    plt.imshow(b, cmap=newcmp, aspect='equal')
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
    ins = 0
    area = 0
    for i, list in enumerate(b_cut):
        for j, value in enumerate(list):
            if value != 0:
                ins += b[i][j]
                area += 1
    if len(contours) != 0:
        counts = (ins / area) - (bg.mean() * a1 + b1)
    else:
        counts = b.mean() - (bg.mean() * a1 + b1)
    return len(contours), counts, b_cut


def find_ions_mor_spec(a, r1, b_cut, a1=1, b1=0):
    """
    find ions by morphology
    :param a: numpy 2-d array
    :param r1: threshold area
    :param r: threshold ratio
    :param a1: a corrected bg factor
    :param b1: b corrected bg factor
    :return: contours lens, mean counts minus bg / plt show in outside, contours， b_cut
    """
    bg = a[r1[1]:(r1[1] + r1[3]), (r1[0] + r1[2]):(r1[0] + r1[2] * 2)]
    b = a[r1[1]:(r1[1] + r1[3]), r1[0]:(r1[0] + r1[2])]
    contours = measure.find_contours(b_cut, 1)
    plt.imshow(b, cmap=newcmp, aspect='equal')
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
    ins = 0
    area = 0
    for i, list in enumerate(b_cut):
        for j, value in enumerate(list):
            if value != 0:
                ins += b[i][j]
                area += 1
    if len(contours) != 0:
        counts = (ins / area) - (bg.mean() * a1 + b1)
    else:
        counts = b.mean() - (bg.mean() * a1 + b1)
    return len(contours), counts


def find_ions_grabcut(file, a):
    """
    find ions by grabcut
    :param file: specific png file name
    :param a: img numpy 2-d array
    :return: rectangle character
    """
    cvpath = file.replace(".csv", ".png")
    img = cv2.imread(cvpath, 1)
    cv2.flip(img, 0, img)
    size = img.shape
    ry = len(a) / size[0]
    rx = len(a[0]) / size[1]
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)  # bg model
    fgdModel = np.zeros((1, 65), np.float64)  # front model
    rect = (5, 5, size[1] - 10, size[0] - 10)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # 获取分割后的信息
    mask_d = cv2.dilate(mask2, None, iterations=4)  # 形态学膨胀
    r0 = cv2.boundingRect(mask_d)  # 形状描述子
    r1 = [int(r0[0] * rx), int(r0[1] * ry), int(r0[2] * rx), int(r0[3] * ry)]
    return r1


# <--------------------------Curve fitting Method----------------->
def f_expdec(x, a, b):
    """
    exp function
    :param x: var
    :param a: const
    :param b: const
    :return: function
    """
    return a * np.exp(-x * b)


def f_1(x, A, B):
    """
    linear function
    :param x: var
    :param A: const
    :param B: const
    :return: function
    """
    return A * x + B


def f_fra(x, A, B):
    """
    fractional function
    :param x: var
    :param A: const
    :param B: const
    :return: function
    """
    return (A * x) / (x - B)


def r_squre(y1, y2):
    """
    r2 get
    :param y1: scatter numpy 1-d array
    :param y2: fitted numpy 1-d array
    :return: r-square value
    """
    SST = sum((y1 - np.mean(y1)) ** 2)
    SSR = sum((y1 - y2) ** 2)
    R_2 = 1 - SSR / SST
    return R_2


def klget(lt, p, t=298.15):
    """
    Calculate the value of Kl
    :param lt: seconds
    :param p: torr
    :param t: K
    :return: kl as cm3/s
    """
    pa = p * 133.3224
    nn = pa / (1.380649e-23 * t)
    kl = 1 / (lt * nn)
    return kl * 1e6


# Calculate the value of Kl and set in kl as cm3/s
kl = lambda lt, p, t: (1 / (lt * (p * 133.3224) / (1.380649e-23 * t))) * 1e6


# <----------------------------Show Method------------------------->
def image_save(local_path):
    """
    save fig after filter as png by multiprocessing
    :param local_path: path as date
    :return: save png
    """
    if (os.path.exists(local_path)):
        print("\n+----------- PNG Saving ...--------+")
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir):  # and "bg" not in dir:
                print("\t\033[1;35mSaving %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                i = 0
                l = len(os.listdir(local_path + "/" + dir))
                filelist = sorted(os.listdir(local_path + "/" + dir))
                for file in filelist:
                    if ".csv" not in file:  # or "BG" in file:
                        filelist.remove(file)
                threads = []
                for file in filelist:
                    t = multiprocessing.Process(target=save_png, args=(file,))
                    threads.append(t)
                    t.start()
                    t.join()
                    i += 1
                    print("\rSaving：{:.2%} .".format(i / l), end="")
                print("\n\t\033[1;35mSaving %s Over!\033[0m" % dir)
        print("\n+--------- PNG Saving Over! --------+")
        os.chdir("../..")
    else:
        print("Dir not exist!!")


def imageshow_all_info(local_path):
    """
    show all image also show max,mean and time
    :param local_path: path as date
    :return:
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imageshowing ----------+")
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir):
                i = 0
                size = 5  # data shown in column
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                length = len(os.listdir(local_path + "/" + dir))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" not in file or "BG" in file:
                        length -= 1
                size_buffer = length % size
                plt.rc("figure", figsize=(10, 4), facecolor="#FFFFFF", dpi=dpi)
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if (length <= size_buffer):
                        size = size_buffer
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        a = np.array(f1)
                        plt.subplot(1, size, i % size + 1)
                        plt.imshow(a,
                                   cmap=newcmp,
                                   aspect='equal',
                                   origin='lower')
                        plt.yticks([])
                        plt.xticks([])
                        plt.title("Max:%d" % a.max(), font)
                        plt.ylabel(Time(name[1], name[2], name[3]), font)
                        plt.xlabel("Mean:%d" % a.mean(), font)
                        if i % size + 1 == size:
                            plt.tight_layout()
                            plt.show()
                            i = -1
                        i += 1
                        length -= 1
                try:
                    plt.show()
                except:
                    pass
                print("\t\033[1;35mRead %s Over!\033[0m" % dir)
        print("\n+---------- imageshowover ----------+")
        os.chdir("../..")
    else:
        print("Dir not exist!!")


def imageshow_all_filter(local_path):
    """
    show after mixfilter as imageshow_all_info
    :param local_path:
    :return:
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imagesaving ----------+")
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir):
                i = 0
                size = 5  # data shown in column
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                length = len(os.listdir(local_path + "/" + dir))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" not in file or "BG" in file:
                        length -= 1
                size_buffer = length % size
                plt.rc("figure", figsize=(10, 4), facecolor="#FFFFFF", dpi=dpi)
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if (length <= size_buffer):
                        size = size_buffer
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        a = np.array(f1)
                        a = image_mixfilter(a)
                        plt.subplot(1, size, i % size + 1)
                        plt.imshow(a, cmap=newcmp, aspect='equal', origin='lower')
                        plt.xticks([])
                        plt.yticks([])
                        plt.title("Max:%d" % a.max())
                        plt.ylabel(Time(name[1], name[2], name[3]), font)
                        plt.xlabel("Mean:%d" % a.mean(), font)
                        if i % size + 1 == size:
                            plt.tight_layout()
                            plt.show()
                            i = -1
                        i += 1
                        length -= 1
                try:
                    plt.show()
                except:
                    pass
                print("\t\033[1;35mRead %s Over!\033[0m" % dir)
        print("\n+---------- imagesaveover ----------+")
        os.chdir("../..")
    else:
        print("Dir not exist!!")


def imageshow_cvions(local_path, subdir="lt"):
    """
    show ions as garbcut, finally show max area filedir
    :param local_path:
    :param subdir: all include chars
    :return:
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imageshowing ----------+")
        maxadd = ""
        maxv = 0
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir) and "bg" not in dir and subdir in dir:
                i = 0
                size = 5  # data shown in column
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                length = len(os.listdir(local_path + "/" + dir))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" not in file or "BG" in file:
                        length -= 1
                size_buffer = length % size
                plt.rc("figure", figsize=(10, 4), facecolor="#FFFFFF", dpi=dpi)
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if (length <= size_buffer):
                        size = size_buffer
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        a = np.array(f1)
                        r1 = find_ions_grabcut(file, a)
                        print(r1)
                        if r1[2] * r1[3] > maxv:
                            maxadd = name[0]
                            maxv = r1[2] * r1[3]
                        ax = plt.subplot(1, size, i % size + 1)
                        plt.imshow(a, cmap=newcmp, aspect='equal', origin='lower')
                        plt.yticks([])
                        plt.xticks([])
                        rectnp = plt.Rectangle((r1[0], r1[1]), r1[2], r1[3], linewidth=2, edgecolor='g',
                                               facecolor='none')
                        ax.add_patch(rectnp)
                        plt.ylabel(Time(name[1], name[2], name[3]), font)
                        if i % size + 1 == size:
                            plt.tight_layout()
                            plt.show()
                            i = -1
                        i += 1
                        length -= 1
                try:
                    plt.show()
                except:
                    pass
                print("\t\033[1;35mRead %s Over!\033[0m" % dir)
        print("\n+---------- imageshowover ----------+")
        print("Max area in: " + maxadd)
        os.chdir("../..")
    else:
        print("Dir not exist!!")


def rect_fitable(local_path, r1, subdir="lt"):
    """
    Show rect fitable?
    :param local_path:
    :param r1: rectangle character as a list[x,y,xs,ys]
    :param subdir: include all input chars
    :return:
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imageshowing ----------+")
        maxadd = ""
        maxv = 0
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir) and "bg" not in dir and subdir in dir:
                i = 0
                size = 5  # data shown in column
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                length = len(os.listdir(local_path + "/" + dir))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" not in file or "BG" in file:
                        length -= 1
                size_buffer = length % size
                plt.rc("figure", figsize=(10, 4), facecolor="#FFFFFF")
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if (length <= size_buffer):
                        size = size_buffer
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        a = np.array(f1)
                        ax = plt.subplot(1, size, i % size + 1)
                        plt.imshow(a, cmap='bone', aspect='equal', origin='lower')
                        plt.xticks([])
                        plt.yticks([])
                        rectnp = plt.Rectangle((r1[0], r1[1]), r1[2], r1[3], linewidth=2, edgecolor='g',
                                               facecolor='none')
                        ax.add_patch(rectnp)
                        plt.ylabel(Time(name[1], name[2], name[3]), font)
                        if i % size + 1 == size:
                            plt.tight_layout()
                            plt.show()
                            i = -1
                        i += 1
                        length -= 1
                try:
                    plt.show()
                except:
                    pass
                print("\t\033[1;35mRead %s Over!\033[0m" % dir)
        print("\n+---------- imageshowover ----------+")
        os.chdir("../..")
    else:
        print("Dir not exist!!")


def imageshow_lt_fit(local_path, r1, subdir=" ", a1=1, b1=0):
    """
    show lt fit select region a,b is bg correction as bg=a*sg+b
    :param local_path: path as date
    :param r1: cut rectangle
    :param subdir: subdir name default empty
    :param a1: bg correct factor a
    :param b1: bg correct factor b
    :return: finally x,y,lt
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imageshowing ----------+")
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir) and "bg" not in dir and subdir in dir:
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                x = np.empty(shape=(0, 0))
                y = np.empty(shape=(0, 0))
                i = 0
                size = 5  # data shown in column
                length = len(os.listdir(local_path + "/" + dir))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" not in file or "BG" in file:
                        length -= 1
                size_buffer = length % size
                plt.rc("figure", figsize=(8, 4), facecolor="#FFFFFF", dpi=dpi)
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if (length <= size_buffer):
                        size = size_buffer
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        ashow = np.array(f1)[r1[1]:(r1[1] + r1[3]), r1[0]:(r1[0] + r1[2])]
                        # a = image_mixfilter(f1)
                        a = np.array(f1)
                        b = a[r1[1]:(r1[1] + r1[3]), r1[0]:(r1[0] + r1[2])]
                        bg = a[r1[1]:(r1[1] + r1[3]), (r1[0] + r1[2]):(r1[0] + r1[2] * 2)]
                        counts = b.mean() - ((bg.mean()) * a1 + b1)
                        ax = plt.subplot(1, size, i % size + 1)
                        plt.imshow(ashow, cmap=newcmp, aspect='equal')
                        plt.xticks([])
                        plt.yticks([])
                        # rectnp = plt.Rectangle((r1[0], r1[1]), r1[2], r1[3], linewidth=2, edgecolor='g', facecolor='none')
                        y = np.append(y, counts)
                        plt.ylabel(Time(name[1], name[2], name[3]), font)
                        x = np.append(x, int(name[1]) * 60 + int(name[2]) + int(name[3]) / 60)
                        plt.xlabel("Counts: %d" % counts, font)
                        if i % size + 1 == size:
                            plt.tight_layout()
                            plt.show()
                            i = -1
                        i += 1
                        length -= 1
                if len(x) != 0:
                    x = x - x[0]
                    plt.scatter(x, y, c='k', marker='.')
                    popt, pcov = curve_fit(f_expdec, x, y)
                    y1 = np.array([f_expdec(k, popt[0], popt[1]) for k in x])
                    plt.plot(x, y1, 'r')
                    plt.xlabel("Time/min", font)
                    plt.ylabel("Counts after filter", font)
                    plt.show()
                    print("R2 is : %.4f" % r_squre(y, y1))
                    print("Life Time: %.4f mins" % (1 / popt[1]))
                    bf = sorted(os.listdir(local_path + "/" + dir))
                    f = []
                    for file in bf:
                        if ".csv" in file and "BG" not in file:
                            f.append(file)
                    nf = [f[0], f[int(len(f) / 4)], f[int(len(f) / 2)], f[int(len(f) / 4 * 3)], f[-1]]
                    name = nf[0].split("_")
                    t0 = int(name[1]) * 3600 + int(name[2]) * 60 + int(name[3])
                    plt.rc("figure", figsize=(6, 4), facecolor="#FFFFFF")
                    for i, file in enumerate(nf):
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        ashow = np.array(f1)[r1[1]:(r1[1] + r1[3]), r1[0]:(r1[0] + r1[2])]
                        ax = plt.subplot(1, 5, i + 1)
                        plt.imshow(ashow, cmap=newcmp, aspect='equal', interpolation='gaussian')
                        name = file.split("_")
                        t = int(name[1]) * 3600 + int(name[2]) * 60 + int(name[3])
                        t = t - t0
                        plt.title("%d m %d s" % (int(t / 60), t % 60), font)
                        plt.axis('off')
                    plt.tight_layout()
                    plt.show()
        print("\n+---------- imageshowover ----------+")
        os.chdir("../..")
        return x, y, (1 / popt[1])
    else:
        print("Dir not exist!!")


def imageshow_morions_fit(local_path, r1, subdir=" ", r=0.3, a1=1, b1=0):
    """
    show ions as morphology
    :param local_path:
    :param r1:
    :param subdir:
    :param r:
    :param a1:
    :param b1:
    :return: b_cut list, x list, y list, err list
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imageshowing ----------+")
        x = np.empty(shape=(0, 0))
        y = np.empty(shape=(0, 0))
        err = np.empty(shape=(0, 0))
        b_cuts = []
        series = 0
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir) and "bg" not in dir and subdir in dir:
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                v = np.empty(shape=(0, 0))
                i = 0
                size = 5  # data shown in column
                length = len(os.listdir(local_path + "/" + dir))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" not in file or "BG" in file:
                        length -= 1
                size_buffer = length % size
                plt.rc("figure", figsize=(8, 4), facecolor="#FFFFFF", dpi=dpi)
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if (length <= size_buffer):
                        size = size_buffer
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        # a = np.array(f1)
                        a = image_mixfilter(f1)
                        plt.subplot(1, size, i % size + 1)
                        n, counts, b_cut = find_ions_mor(a, r1, r, a1, b1)
                        b_cuts.append(b_cut)
                        plt.xticks([])
                        plt.yticks([])
                        plt.title("%d Ions" % n, font)
                        series += 1
                        plt.ylabel(Time(name[1], name[2], name[3]), font)
                        plt.xlabel("Counts:%d" % counts, font)
                        v = np.append(v, counts)
                        if i % size + 1 == size:
                            plt.tight_layout()
                            plt.show()
                            print("right index:" + str(series - 1))
                            i = -1
                        i += 1
                        length -= 1
                try:
                    plt.show()
                except:
                    pass
                print("Mean Counts: " + str(v.mean()))
                print("Stand Err: " + str(np.std(v, ddof=1)))
                x = np.append(x, float(re.findall(r"\d+\.?\d*", name[0].replace("-", "."))[0]))
                y = np.append(y, v.mean())
                err = np.append(err, np.std(v, ddof=1))
        print("\n+---------- imageshowover ----------+")
        os.chdir("../..")
        xy = np.vstack((x, y, err))
        xy = xy.T[np.lexsort(xy[::-1, :])].T
        x = xy[0]
        y = xy[1]
        err = xy[2]
        A, B = curve_fit(f_fra, x, y)[0]
        x_fit = np.arange(0, x.max() * 2, 1)
        y_fit = f_fra(x_fit, A, B) / abs(2 * A)
        y = y / abs(2 * A)
        err = err / abs(2 * A)
        plt.errorbar(x, y, yerr=err, fmt='ro', ecolor='k', elinewidth=2, capsize=4)
        plt.title(subdir)
        plt.ylabel("The Excited State Population", font)
        plt.xlabel(" Optical power(mW)", font)
        plt.plot(x_fit, y_fit, 'r', linewidth=1, label='fitted curve')
        plt.legend(loc=2, framealpha=0.2)
        plt.show()
        print("A:%.4f, B:%.4f" % (abs(A), abs(B)))
        y_comp = np.array([f_fra(k, A, B) for k in x]) / abs(2 * A)
        print("R2 is : %.4f" % r_squre(y, y_comp))
    else:
        print("Dir not exist!!")
    return b_cuts, x.tolist(), y.tolist(), err.tolist()


def imageshow_morions_spec_fit(local_path, r1, b_cut, subdir=" ", a1=1, b1=0):
    """
    (reserved) show ions as morphology with spec area
    :param local_path:
    :param r1:
    :param subdir:
    :param r:
    :param a1:
    :param b1:
    :return:
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imageshowing ----------+")
        x = np.empty(shape=(0, 0))
        y = np.empty(shape=(0, 0))
        err = np.empty(shape=(0, 0))
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir) and "bg" not in dir and subdir in dir:
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                v = np.empty(shape=(0, 0))
                i = 0
                size = 5  # data shown in column
                length = len(os.listdir(local_path + "/" + dir))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" not in file or "BG" in file:
                        length -= 1
                size_buffer = length % size
                plt.rc("figure", figsize=(8, 4), facecolor="#FFFFFF", dpi=dpi)
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if (length <= size_buffer):
                        size = size_buffer
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        # a = np.array(f1)
                        a = image_mixfilter(f1)
                        plt.subplot(1, size, i % size + 1)
                        n, counts = find_ions_mor_spec(a, r1, b_cut, a1, b1)
                        plt.xticks([])
                        plt.yticks([])
                        # plt.title("%d Ions" % n, font)
                        plt.ylabel(Time(name[1], name[2], name[3]), font)
                        plt.xlabel("Counts:%d" % counts, font)
                        v = np.append(v, counts)
                        if i % size + 1 == size:
                            plt.tight_layout()
                            plt.show()
                            i = -1
                        i += 1
                        length -= 1
                try:
                    plt.show()
                except:
                    pass
                print("Mean Counts: " + str(v.mean()))
                print("Stand Err: " + str(np.std(v, ddof=1)))
                x = np.append(x, float(re.findall(r"\d+\.?\d*", name[0].replace("-", "."))[0]))
                y = np.append(y, v.mean())
                err = np.append(err, np.std(v, ddof=1))
        print("\n+---------- imageshowover ----------+")
        os.chdir("../..")
        plt.errorbar(x, y, yerr=err, fmt='go', ecolor='r', elinewidth=2, capsize=4)
        plt.ylabel("Mean Counts", font)
        plt.xlabel(subdir + " arrange", font)
        plt.xlim(0, None)
        plt.show()
    else:
        print("Dir not exist!!")


def bg_fig(local_path, r1):
    """
    bg correction, finall show correct a and b
    :param local_path:
    :param r1:
    :return: A1, B1
    """
    if (os.path.exists(local_path)):
        os.chdir(local_path)
        x = np.empty(shape=(0, 0))
        y = np.empty(shape=(0, 0))
        for file in sorted(os.listdir(local_path)):
            if ".csv" in file:
                name = file.split("_")
                f1 = pd.read_csv(file, header=None)
                f1 = f1.dropna()
                a = np.array(f1)
                sa = a[r1[1]:(r1[1] + r1[3]), r1[0]:(r1[0] + r1[2])]
                sb = a[r1[1]:(r1[1] + r1[3]), (r1[0] + r1[2]):(r1[0] + r1[2] * 2)]
                x = np.append(x, sb.mean())
                y = np.append(y, sa.mean())
        os.chdir("../..")
        A1, B1 = curve_fit(f_1, x, y)[0]
        y1 = np.array([f_1(k, A1, B1) for k in x])
        # ax = plt.axes()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        plt.plot(x, y1, 'r')
        plt.scatter(x, y, c='g', marker='x')
        plt.ylabel("selection ions bg counts", font)
        plt.xlabel("substraction bg counts", font)
        plt.show()
        print("y = " + str(A1) + " * x + " + str(B1))
        print(r_squre(y, y1))
    else:
        print("Dir not exist!!")
    return A1, B1


def imageshow_hist(local_path):
    """
    show hist finally stats
    :param local_path:
    :return:
    """
    if (os.path.exists(local_path)):
        print("\n+---------- imageshowing ----------+")
        for dir in sorted(os.listdir(local_path)):
            if os.path.isdir(local_path + "/" + dir):
                print("\t\033[1;35mReading %s ...\033[0m" % dir)
                os.chdir(local_path + "/" + dir)
                x = np.empty(shape=(0, 0))
                y = np.empty(shape=(0, 0))
                for file in sorted(os.listdir(local_path + "/" + dir)):
                    if ".csv" in file:
                        name = file.split("_")
                    if ".csv" in file and "BG" not in file:
                        f1 = pd.read_csv(file, header=None)
                        f1 = f1.dropna()
                        a = np.array(f1)
                        a = image_mixfilter(a)
                        n, bins = get_hist(a)
                        # n, bins, patches = plt.hist(b_cv.ravel(), 256)
                        # plt.title("Max:%d" % a.max())
                        # plt.ylabel(Time(name[1],name[2],name[3]))
                        # plt.xlabel("hist Max:%d" %bins[np.unravel_index(int(n.argmax()), n.shape)])
                        # plt.show()
                        y = np.append(y, bins[np.unravel_index(int(n.argmax()), n.shape)])
                        x = np.append(x, int(name[1]) * 60 + int(name[2]) + int(name[3]) / 60)
                if len(x) != 0:
                    x = x - x[0]
                    plt.plot(x, y, 'r', linewidth=0.5)
                    plt.show()
                print("\t\033[1;35mRead %s Over!\033[0m" % dir)
        print("\n+---------- imageshowover ----------+")
        os.chdir("../..")
    else:
        print("Dir not exist!!")

# <---------------------------------------------------------------->

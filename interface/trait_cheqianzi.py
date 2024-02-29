"""
@author : hogan
time:
function: 酸枣仁栅状细胞侧面观显微结构检测界面
"""
import tkinter as tk
from tkinter import DISABLED, NORMAL, StringVar, Label, Button, filedialog, PhotoImage
from tkinter.ttk import Progressbar
import datetime
import torch
from functions.micro_detect import detect, detect_one
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk

import os
import sys

result_dict = {'file_dict': "", 'type': 0, 'res_img': None, 'name': "", 'save_path': ""} # 存储全局变量

# 资源文件目录访问
def source_path(relative_path):
    # 是否Bundle Resource
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath("..")
    return os.path.join(base_path, relative_path)


# 修改当前工作目录，使得资源文件可以被正确访问
cd = source_path('')
os.chdir(cd)

def thread_it(func, *args):
    '''将函数打包进线程'''
    # 创建
    t = threading.Thread(target=func, args=args)
    # 守护 !!!
    t.setDaemon(True)
    # 启动
    t.start()
    # 阻塞--卡死界面！
    # t.join()

########################################## 批量检测组件函数
# 批量检测选择待测文件夹
def select_folder(var, label, button, window):
    button['state'] = DISABLED
    button['text'] = '选择中'

    folder_path = filedialog.askdirectory()
    selected_path = folder_path
    var.set('读取自: ' + selected_path)
    label.place(relx=0.04, rely=0.22)
    window.update()
    result_dict['file_dict'] = selected_path

    button['state'] = NORMAL
    button['text'] = '待测图片文件夹'

# 批量检测的函数
def detect_batch(indir, outdir, button, bar, window):
    button['state'] = DISABLED
    button['text'] = '检测中'

    var = StringVar()
    l = Label(window, textvariable=var, bg="#f0f0f0", font=('宋体', 8), width=100, height=1)

    if indir == "":
        var.set('输入路径为空')
        l.place(relx=0.04, rely=0.67)
        window.update()
        button['state'] = NORMAL
        button['text'] = '开始检测'
        return
    indir = indir.replace('\\', '/')
    if not os.path.exists(indir) or not os.path.isabs(indir): # 检查路径是否存在
        var.set('输入路径不存在')
        l.place(relx=0.04, rely=0.67)
        window.update()
        button['state'] = NORMAL
        button['text'] = '开始检测'
        return
    if not os.path.isdir(indir): # 检查路径是否是一个文件夹
        var.set('输入路径不是一个文件夹')
        l.place(relx=0.04, rely=0.67)
        window.update()
        button['state'] = NORMAL
        button['text'] = '开始检测'
        return

    if outdir == "":
        var.set('保存路径为空')
        l.place(relx=0.04, rely=0.67)
        window.update()
        button['state'] = NORMAL
        button['text'] = '开始检测'
        return
    outdir = outdir.replace('\\', '/') + '/'
    if not os.path.exists(outdir) or not os.path.isabs(outdir):  # 检查路径是否存在
        var.set('保存路径不存在')
        l.place(relx=0.04, rely=0.67)
        window.update()
        button['state'] = NORMAL
        button['text'] = '开始检测'
        return
    if not os.path.isdir(outdir): # 检查路径是否是一个文件夹
        var.set('保存路径不是一个文件夹')
        l.place(relx=0.04, rely=0.67)
        window.update()
        button['state'] = NORMAL
        button['text'] = '开始检测'
        return

    foldername = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') # 获取系统时间(年月日时分秒)为文件名
    # 检查输出文件夹是否已经存在，如果已经存在则加i直到不冲突
    if not os.path.exists(outdir + foldername):
        os.makedirs(outdir + foldername)
    else:
        i = 2
        while os.path.exists(outdir + foldername + '_' + str(i)):
            i += 1
        foldername += ('_' + str(i))
        os.makedirs(outdir + foldername)

    # 给模型输入输出路径等参数，开始检测
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")  # 系统能提供的device

    # cmd调用模型，传入--device，输入路径--source，输出文件夹集--project，具体的文件夹名--name
    var.set('检测中...')
    l.place(relx=0.04, rely=0.67)
    window.update()

    detect('side', bar, window, device, indir, outdir, foldername, 'micro_stage1.pt', './micro/side4_02_resnet50_1.pkl')

    var.set('检测完成')
    l.place(relx=0.04, rely=0.67)
    window.update()

    button['state'] = NORMAL
    button['text'] = '开始检测'

    # 直接打开结果所在文件夹
    fp = outdir + foldername
    fp: str = fp.replace("/", "\\")
    os.startfile(fp)

def select_save(var, label, button, window):
    button['state'] = DISABLED
    button['text'] = '选择中'

    path = filedialog.askdirectory()
    var.set('保存至: ' + path)
    label.place(relx=0.04, rely=0.44)
    var = StringVar()
    var.set('将在此路径下以检测开始时间创建子文件夹用以保存结果')
    l = Label(window, textvariable=var, bg="#f0f0f0", font=('宋体', 8), width=100, height=1)
    l.place(relx=0.04, rely=0.5)
    window.update()
    result_dict['save_path'] = path

    button['state'] = NORMAL
    button['text'] = '保存路径'
############################################################# 批量检测组件函数结束

############################################################### 单张检测组件函数
# 选择读取图片并检测
def select_img(var, label, window, button, button2):
    button['state'] = DISABLED
    button['text'] = '选择中'

    selected_path = ""
    file_path = filedialog.askopenfilename()
    if file_path:
        selected_path = file_path
    if selected_path == "":
        var.set('路径为空')
        label.place(relx=0.02, rely=0.15)
        window.update()
        button['state'] = NORMAL
        button['text'] = '待测图片'
        return
    if selected_path[-3:].lower() not in ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']:
        var.set('所选文件不是一张图片')
        label.place(relx=0.02, rely=0.15)
        window.update()
        button['state'] = NORMAL
        button['text'] = '待测图片'
        return

    # 开始检测
    button['text'] = '检测中'
    img = cv2.imdecode(np.fromfile(selected_path, dtype=np.uint8), -1)
    if img.shape[-1] > 3:
        img = img[:, :, 0:3].astype(np.uint8)
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")  # 系统能提供的device
    res = detect_one('side', img, device, 'micro_stage1.pt', './micro/side4_02_resnet50_1.pkl')
    result_dict['res_img'] = res
    result_dict['name'] = file_path.split('/')[-1]
    # 展示结果
    # res = cv2.resize(res, (res.shape[1] // 3, res.shape[0] // 3))
    # cv2.imshow('result', res)
    # cv2.waitKey()
    var.set('检测完成，请选择保存路径')
    label.place(relx=0.02, rely=0.15)
    window.update()

    button['state'] = NORMAL
    button['text'] = '选择图片'
    button2['state'] = NORMAL
    button2['text'] = '保存图片'

    # 展示结果
    res.tight_layout()
    res.savefig('./res.png', dpi=300, bbox_inches='tight')
    img =  cv2.imread("./res.png")
    img = cv2.resize(img, (850, 650))
    cv2.imwrite('./res.png', img)

    global label_img
    img = Image.open('../res.png')
    label_img = ImageTk.PhotoImage(img)
    l0 = Label(window,
            image=label_img,  # 标签的文字
            # bg="#f0f0f0",  # 标签背景颜色
            bg="#ffffff",  # 标签背景颜色
            # width=500, height=300  # 标签长宽
            )
    # l0.pack()
    l0.place(relx=0.17, rely=0.2)
    window.update()

# 保存图片按钮函数
def save_img(window, button):
    button['state'] = DISABLED
    button['text'] = '保存中'

    folder_path = filedialog.asksaveasfilename(title=u'保存', initialfile=result_dict['name'], filetypes=[("JPG图片", ".jpg"), ("PNG图片", ".png"), ("JPEG图片", ".jpeg"), ("TIF图片", ".tif")])

    var = StringVar()
    l = Label(window, textvariable=var, bg="#f0f0f0", font=('宋体', 8), width=100, height=1)

    selected_path = folder_path
    result_dict['res_img'].savefig(selected_path, dpi=300, bbox_inches='tight')
    # cv2.imencode(result_dict['name'][-4:], result_dict['res_img'])[1].tofile(selected_path)
    var.set('保存结果至' + selected_path)
    l.place(relx=0.04, rely=0.44)
    window.update()
    os.startfile(folder_path)

    button['state'] = NORMAL
    button['text'] = '保存'
############################################################### 单张检测组件函数结束

def change_type(change, batch_objs, single_objs, window): ################################################# 重写转换函数，要正确摆放位置，并且要清除过时的label
    ori = window.winfo_children()
    if result_dict['type'] == 0: # 换为单张检测
        result_dict['type'] = 1
        change['text'] = '换为批量检测'

        # 删除批量检测各个组件
        for obj in ori:
            obj.place_forget()

        # 布置单张检测组件
        img = single_objs['img']
        save = single_objs['save']
        title = single_objs['title']
        # l0 = single_objs['label']
        change.place(relx=0.015, rely=0.022)
        title.place(relx=0.4, rely=0.01)
        img.place(relx=0.25, rely=0.1)
        save.place(relx=0.75, rely=0.1)
        # l0.place(relx=0.27, rely=0.77)

    else: # 换为批量检测
        result_dict['type'] = 0
        change['text'] = '换为单张检测'

        # 删除单张检测各个组件
        for obj in ori:
            obj.place_forget()

        # 布置批量检测各个组件
        folder = batch_objs['folder']
        save_folder = batch_objs['save_folder']
        start_detect = batch_objs['start_detect']
        title = single_objs['title']
        # l0 = batch_objs['label']
        change.place(relx=0.015, rely=0.022)
        title.place(relx=0.42, rely=0.01)
        folder.place(relx=0.42, rely=0.22)
        save_folder.place(relx=0.42, rely=0.44)
        start_detect.place(relx=0.42, rely=0.66)
        # l0.place(relx=0.27, rely=0.77)

def show_interface_4():
    interface_4_window = tk.Toplevel()
    interface_4_window.title("酸枣仁栅状细胞侧面观显微结构检测")
    interface_4_window.geometry('1200x800')
    interface_4_window.resizable(False, False)  # 禁止最大化

    ########################################################### 批量检测组件声明
    # lable，说明输入框作用
    var1 = StringVar()
    l1 = Label(interface_4_window,
               textvariable=var1,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 8),  # 字体和字体大小
               width=100, height=1  # 标签长宽
               )
    # 进度条
    # bar = tkinter.ttk.Progressbar(window)
    bar = Progressbar(interface_4_window)
    # 按钮，点击则开始一次批量检测
    start_detect = Button(interface_4_window, text='开始检测', width=15, height=2,
                          command=lambda: thread_it(detect_batch, result_dict['file_dict'], result_dict['save_path'],
                                                    start_detect, bar, interface_4_window))  # 点击按钮执行一# 个名为“hit_me”的函数
    # 按钮，选择待测图片所在文件夹
    folder = Button(interface_4_window, text="待测图片文件夹", width=15, height=2,
                    command=lambda: thread_it(select_folder, var1, l1, folder, interface_4_window))
    # lable，说明输入框作用
    var2 = StringVar()
    l5 = Label(interface_4_window,
               textvariable=var2,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 8),  # 字体和字体大小
               width=100, height=1  # 标签长宽
               )
    # 选择保存路径的按钮
    save_folder = Button(interface_4_window, text="保存路径", width=15, height=2,
                         command=lambda: thread_it(select_save, var2, l5, save_folder, interface_4_window))

    # label，展示标签
    label_img = PhotoImage(file='../imgs/labels.png')
    l0 = Label(interface_4_window,
               image=label_img,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               width=300, height=100  # 标签长宽
               )
    #################################################### 批量检测组件声明结束

    ############################################################### 单张检测组件声明
    # 选择图片的标签
    var3 = StringVar()
    l3 = Label(interface_4_window,
               textvariable=var3,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 10),  # 字体和字体大小
               width=90, height=1  # 标签长宽
               )

    title = Label(interface_4_window,
                  text="酸枣仁栅状细胞侧面观显微结构检测",  # 标签的文字
                  bg="#f0f0f0",  # 标签背景颜色
                  font=('宋体', 16),  # 字体和字体大小
                  width=30, height=3  # 标签长宽
                  )

    # 保存图片按钮
    save = Button(interface_4_window, text="保存", width=15, height=2, font=('宋体', 10),
                  command=lambda: thread_it(save_img, interface_4_window, save))
    save['state'] = DISABLED

    # 选择图片并检测的按钮
    img = Button(interface_4_window, text="待测图片", width=15, height=2, font=('宋体', 10),
                 command=lambda: thread_it(select_img, var3, l3, interface_4_window, img, save))
    ############################################################### 单张检测组件声明结束

    # 切换单张检测和批量检测
    batch_objs = {'folder': folder, 'save_folder': save_folder, 'start_detect': start_detect, 'title': title}
    single_objs = {'img': img, 'save': save, 'title': title}
    change = Button(interface_4_window, text="换为单张检测", width=10, height=1, font=('宋体', 10),
                    command=lambda: thread_it(change_type, change, batch_objs, single_objs, interface_4_window))
    change.place(relx=0.015, rely=0.022)

    ############################################################## 批量检测组件布置
    # folder.place(x=275, y=50)
    title.place(relx=0.42, rely=0.01)
    folder.place(relx=0.42, rely=0.22)
    save_folder.place(relx=0.42, rely=0.44)
    start_detect.place(relx=0.42, rely=0.66)
    # l0.place(relx=0.27, rely=0.77)
    ############################################################### 批量检测组件布置结束

    # ############################################################### 单张检测组件布置
    # l3.pack()
    # img.pack()
    # save.pack()
    # scroll2.pack(side="right", fill=tk.Y) # 放在最右侧
    # scroll2.config(command=logs2.yview) # 滚动条与文本框关联
    # logs2.config(yscrollcommand=scroll2.set) # 文本框与滚动条关联
    # l4.pack()
    # logs2.pack()
    # ############################################################### 单张检测组件布置结束

    # 进入消息循环
    interface_4_window.mainloop()
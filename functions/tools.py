"""
@author : hogan
time:
function: some basic functions used for UI and threading purposes
"""
import tkinter as tk
from tkinter import DISABLED, NORMAL, StringVar, Label, Button, filedialog
import os
import sys
import threading

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

# 资源文件目录访问
def source_path(relative_path):
    # 是否Bundle Resource
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

########################################## 批量检测组件函数
# 批量检测选择待测文件夹
def select_folder(var, label, button, window, result_dict):
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


def select_save(var, label, button, window, result_dict):
    button['state'] = DISABLED
    button['text'] = '选择中'

    path = filedialog.askdirectory()
    var.set('保存至: ' + path)
    label.place(relx=0.57, rely=0.49)
    # l = Label(window, textvariable=var, bg="#f0f0f0", font=('宋体', 8), width=100, height=1)
    # l.place(relx=0.67, rely=0.44)
    window.update()
    result_dict['save_path'] = path

    button['state'] = NORMAL
    button['text'] = '保存路径'
############################################################# 批量检测组件函数结束

############################################################### 单张检测组件函数
# 保存图片按钮函数
def save_img(window, button, result_dict):
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


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
from PIL import Image, ImageTk


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
    # var.set('读取自: ' + selected_path)
    # label.place(relx=0.04, rely=0.32)
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

    folder_path = filedialog.asksaveasfilename(title=u'保存', initialfile=result_dict['name'],
                                               filetypes=[("JPG图片", ".jpg"), ("PNG图片", ".png"),
                                                          ("JPEG图片", ".jpeg"), ("TIF图片", ".tif")])

    var = StringVar()
    l = Label(window, textvariable=var, bg="#f0f0f0", font=('宋体', 8), width=100, height=1)

    selected_path = folder_path
    result_dict['res_img'].savefig(selected_path, dpi=300, bbox_inches='tight')
    # cv2.imencode(result_dict['name'][-4:], result_dict['res_img'])[1].tofile(selected_path)
    var.set('保存结果至' + selected_path)
    l.place(relx=0.24, rely=0.44)
    window.update()
    os.startfile(folder_path)

    button['state'] = NORMAL
    button['text'] = '保存'


############################################################### 单张检测组件函数结束

# 用于按钮显示
def load_image(os_path,  filename, type):
    filename = os.path.join(os_path, filename)
    # print("buton img path:{}".format(filename))
    image = Image.open(filename)
    image = image.convert("RGBA")
    if type == 'main':
        image.thumbnail((240, 60))
    elif type == 'micro' or type == 'trait':
        image.thumbnail((600, 140))
    elif type == 'small':
        image.thumbnail((160, 50))
    elif type == 'title':
        image.thumbnail((300, 70))
    else:
        image.thumbnail((240, 60))
    photo = ImageTk.PhotoImage(image)

    return photo

# 接口界面展示背景图，os_path以项目路径为起始路径
def load_bg(window, os_path, path):
    window.update()
    bg_path = os.path.join(os_path, path)
    bg_image = Image.open(bg_path)
    # print(bg_path)
    # print(window.winfo_width(), window.winfo_height())
    bg_image = bg_image.resize((window.winfo_width(), window.winfo_height()))
    bg_photo = ImageTk.PhotoImage(bg_image)
    window.image = bg_photo
    # 创建一个 Canvas
    canvas = tk.Canvas(window, width=window.winfo_width(), height=window.winfo_height())
    canvas.pack(fill="both", expand=True)
    # 在 Canvas 上显示背景图像
    canvas.create_image(0, 0, anchor="nw", image=window.image)
    # # canvas.tag_lower(bg_photo)  # 将图像放置在最底层

# 清除窗口组件，用于窗口刷新
def clear_window(window):
    window.after(0, lambda: [widget.destroy() for widget in window.winfo_children()])

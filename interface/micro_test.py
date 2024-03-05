"""
@author : hogan
time:
function:
"""
from interface import micro_origin, micro
import tkinter as tk
from PIL import Image, ImageTk
import os
from functions import tools

def open_interface_1(window):
    micro_south.show_interface_1(window)

def open_interface_2(window):
    micro_suanzaoren.show_interface_2(window)

def show_micro_window():
    # global window
    # 创建选择主界面
    window = tk.Toplevel()
    window.title("中药材显微结构检测")
    # 设置界面内容
    window.geometry('1000x750')
    window.resizable(False, False)

    # 设置背景图片
    os_path = os.path.dirname(__file__)
    os_path = os.path.dirname(os_path) # 返回上级目录

    tools.load_bg(window, os_path, 'background/detect.png')

    button1_image = tools.load_image(os_path, 'button/micro_south_button.png', 'micro')
    button2_image = tools.load_image(os_path, 'button/micro_suanzaoren_button.png', 'micro')

    # 创建按钮1，指向界面1
    button1 = tk.Button(window, image=button1_image, width=400, height=75, command=lambda: open_interface_1(window))
    button1.pack(padx=20, pady=10)

    # 创建按钮2，指向界面2
    button2 = tk.Button(window, image=button2_image, width=400, height=75, command=lambda: open_interface_2(window))
    button2.pack(padx=20, pady=10)

    button1.place(relx=0.32, rely=0.25)
    button2.place(relx=0.32, rely=0.45)

    window.mainloop()

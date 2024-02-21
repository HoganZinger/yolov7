"""
@author : hogan
time:
function: 酸枣仁显微结构检测界面
"""
import tkinter as tk
from tkinter import DISABLED, NORMAL, StringVar, Label, Button, filedialog, PhotoImage
from tkinter import ttk
from tkinter.ttk import Progressbar, Combobox
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
from functions.detect import detect_2, detect_one_2
from functions.tools import thread_it, source_path, select_folder, save_img
from functions.review import set_combobox_placeholder, combobox_set_normal_style, refresh_listbox,  \
     refresh_result, change_selection, show_image, result_list_event, result_select_event, show_report_window,  \
     refresh_combobox


result_dict = {'file_dict': "", 'type': 0, 'res_img': None, 'name': "", 'save_path': ""} # 存储全局变量
pre_list = []
path_list = []
res_img_list = []

# 修改当前工作目录，使得资源文件可以被正确访问
cd = source_path('')
os.chdir(cd)

# 批量检测的函数
def detect_batch(indir, outdir, button, bar, window, results_list, result_select):
    global path_list
    global pre_list
    global res_img_list
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

    # if outdir == "":
    #     #     var.set('保存路径为空')
    #     #     l.place(relx=0.04, rely=0.67)
    #     #     window.update()
    #     #     button['state'] = NORMAL
    #     #     button['text'] = '开始检测'
    #     #     return
    #     # outdir = outdir.replace('\\', '/') + '/'
    # if not os.path.exists(outdir) or not os.path.isabs(outdir):  # 检查路径是否存在
    #     var.set('保存路径不存在')
    #     l.place(relx=0.04, rely=0.67)
    #     window.update()
    #     button['state'] = NORMAL
    #     button['text'] = '开始检测'
    #     return
    # if not os.path.isdir(outdir): # 检查路径是否是一个文件夹
    #     var.set('保存路径不是一个文件夹')
    #     l.place(relx=0.04, rely=0.67)
    #     window.update()
    #     button['state'] = NORMAL
    #     button['text'] = '开始检测'
    #     return
    #
    # foldername = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') # 获取系统时间(年月日时分秒)为文件名
    # # 检查输出文件夹是否已经存在，如果已经存在则加i直到不冲突
    # if not os.path.exists(outdir + foldername):
    #     os.makedirs(outdir + foldername)
    # else:
    #     i = 2
    #     while os.path.exists(outdir + foldername + '_' + str(i)):
    #         i += 1
    #     foldername += ('_' + str(i))
    #     os.makedirs(outdir + foldername)

    # 给模型输入输出路径等参数，开始检测
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")  # 系统能提供的device

    # cmd调用模型，传入--device，输入路径--source，输出文件夹集--project，具体的文件夹名--name
    var.set('检测中...')
    l.place(relx=0.04, rely=0.67)
    window.update()

    pre_list, path_list = detect_2(bar, window, device, indir, 'micro_stage1.pt', './pickles/surface_02_resnet50_1.pkl',
                            './pickles/side4_02_resnet50_1.pkl', './pickles/neizhongpi_01_resnet50_1.pkl')

    var.set('检测完成')
    l.place(relx=0.04, rely=0.67)
    window.update()

    button['state'] = NORMAL
    button['text'] = '开始检测'

    refresh_listbox(results_list, pre_list)
    refresh_combobox(result_select, pre_list)

    # # 直接打开结果所在文件夹
    # fp = outdir + foldername
    # fp: str = fp.replace("/", "\\")
    # os.startfile(fp)

# 单张检测场景中，选择读取图片并检测
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
    res, pre_list = detect_one_2(img, device, 'micro_stage1.pt', './pickles/surface_02_resnet50_1.pkl',
                            './pickles/side4_02_resnet50_1.pkl', './pickles/neizhongpi_01_resnet50_1.pkl')
    result_dict['res_img'] = res
    result_dict['name'] = file_path.split('/')[-1]

    var.set('检测完成，请选择保存路径')
    label.place(relx=0.02, rely=0.15)
    window.update()

    button['state'] = NORMAL
    button['text'] = '选择图片'
    button2['state'] = NORMAL
    button2['text'] = '保存图片'

    # 展示结果
    res.tight_layout()
    res.savefig('../res.png', dpi=300, bbox_inches='tight')
    img =  cv2.imread("../res.png")
    img = cv2.resize(img, (850, 650))
    cv2.imwrite('../res.png', img)

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

# 修改单张和批量检测
def change_type(change, batch_objs, single_objs, window):
    ################################################# 重写转换函数，要正确摆放位置，并且要清除过时的label
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
        # save_folder = batch_objs['save_folder']
        start_detect = batch_objs['start_detect']
        title = single_objs['title']
        results_list = batch_objs['results_list']
        result_select = batch_objs['result_select']
        category_select = batch_objs['category_select']
        confirm = batch_objs['confirm']
        prev_result = batch_objs['prev_result']
        next_result = batch_objs['next_result']
        report = batch_objs['report']
        # l0 = batch_objs['label']
        change.place(relx=0.015, rely=0.022)
        title.place(relx=0.42, rely=0.01)
        folder.place(relx=0.22, rely=0.12)
        # save_folder.place(relx=0.42, rely=0.44)
        start_detect.place(relx=0.22, rely=0.36)
        # l0.place(relx=0.27, rely=0.77)
        results_list.place(relx=0.72, rely=0.12)
        result_select.place(relx=0.42, rely=0.44)
        category_select.place(relx=0.42, rely=0.64)
        confirm.place(relx=0.72, rely=0.64)
        prev_result.place(relx=0.22, rely=0.84)
        next_result.place(relx=0.82, rely=0.84)
        report.place(relx=0.50, rely=0.80)
        

def show_interface_2():
    interface_2_window = tk.Toplevel()
    interface_2_window.title("酸枣仁显微结构检测")
    interface_2_window.geometry('1200x800')
    interface_2_window.resizable(False, False)  # 禁止最大化

    ########################################################### 批量检测组件声明
    # lable，说明输入框作用
    var1 = StringVar()
    l1 = Label(interface_2_window,
               textvariable=var1,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 8),  # 字体和字体大小
               width=100, height=1  # 标签长宽
               )
    # 进度条
    # bar = tkinter.ttk.Progressbar(window)
    bar = Progressbar(interface_2_window)
    # 按钮，点击则开始一次批量检测
    start_detect = Button(interface_2_window, text='开始检测', width=15, height=2,
                          command=lambda: thread_it(detect_batch, result_dict['file_dict'], result_dict['save_path'],
                                                    start_detect, bar, interface_2_window, results_list, result_select))  # 点击按钮执行一# 个名为“hit_me”的函数

    # 按钮，选择待测图片所在文件夹
    folder = Button(interface_2_window, text="待测图片文件夹", width=15, height=2,
                    command=lambda: thread_it(select_folder, var1, l1, folder, interface_2_window, result_dict))
    # lable，说明输入框作用
    var2 = StringVar()
    l5 = Label(interface_2_window,
               textvariable=var2,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 8),  # 字体和字体大小
               width=100, height=1  # 标签长宽
               )
    # # 选择保存路径的按钮
    # save_folder = Button(interface_2_window, text="保存路径", width=15, height=2,
    #                      command=lambda: thread_it(select_save, var2, l5, save_folder, interface_2_window))

    # label，展示标签
    label_img = PhotoImage(file='./imgs/labels.png')
    l0 = Label(interface_2_window,
               image=label_img,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               width=300, height=100  # 标签长宽
               )

    # 结果展示列表
    results_list = tk.Listbox(interface_2_window, width=35, height=10)
    results_list.bind("<<ListboxSelect>>", lambda event, window=interface_2_window: result_list_event(event, window, path_list))
    results_list.pack(padx=10, pady=10)

    # 选择要修改的结果
    placeholder1 = "请选择一条结果"
    style1 = ttk.Style()
    style1.configure('Placeholder.TCombobox', foreground='gray')
    result_select = ttk.Combobox(interface_2_window, state="readonly")
    # result_select.bind("<<ComboboxSelected>>", lambda event, window=interface_2_window, combobox=result_select,
    #                                         listbox=results_list: result_select_event(event, window, combobox, listbox))
    result_select['values'] = pre_list
    result_select.pack(pady=10)
    # 选择要修正为的类别
    placeholder2 = "请选择类别"
    style2 = ttk.Style()
    style2.configure('Placeholder.TCombobox', foreground='gray')
    category_select = Combobox(interface_2_window, state="readonly")
    category_select['values'] = ('草酸钙', '草酸钙（暗）', '酸枣仁内种皮细胞', '理枣仁内种皮细胞', '酸枣仁栅状细胞表面观', '兵豆栅状细胞表面观', '理枣仁栅状细胞表面观'
                                 , '酸枣仁栅状细胞侧面观', '兵豆栅状细胞侧面观', '理枣仁栅状细胞侧面观')
    category_select.pack(pady=10)
    # 设置样式和占位符
    set_combobox_placeholder(result_select, placeholder1)
    set_combobox_placeholder(category_select, placeholder2)

    confirm = Button(interface_2_window, text='确认', width=8, height=2, font=('宋体', 10),
                     command=lambda : thread_it(refresh_result, pre_list, category_select, results_list, result_select))
    prev_result = Button(interface_2_window, text='上一条', width=8, height=2, font=('宋体', 10),
                         command=lambda : thread_it(change_selection, result_select, 'prev'))
    next_result = Button(interface_2_window, text='下一条', width=8, height=2, font=('宋体', 10),
                         command=lambda : thread_it(change_selection, result_select, 'next'))
    report = Button(interface_2_window, text='生成检测报告', width=15, height=3, font=('宋体', 10),
                         command=lambda : thread_it(show_report_window, interface_2_window, report, result_dict,
                                                    pre_list, path_list))
    #################################################### 批量检测组件声明结束

    ############################################################### 单张检测组件声明
    # 选择图片的标签
    var3 = StringVar()
    l3 = Label(interface_2_window,
               textvariable=var3,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 10),  # 字体和字体大小
               width=90, height=1  # 标签长宽
               )

    title = Label(interface_2_window,
                  text="酸枣仁显微结构检测",  # 标签的文字
                  bg="#f0f0f0",  # 标签背景颜色
                  font=('宋体', 16),  # 字体和字体大小
                  width=30, height=3  # 标签长宽
                  )

    # 保存图片按钮
    save = Button(interface_2_window, text="保存", width=15, height=2, font=('宋体', 10),
                  command=lambda: thread_it(save_img, interface_2_window, save, result_dict))
    save['state'] = DISABLED

    # 选择图片并检测的按钮
    img = Button(interface_2_window, text="待测图片", width=15, height=2, font=('宋体', 10),
                 command=lambda: thread_it(select_img, var3, l3, interface_2_window, img, save, result_dict))
    ############################################################### 单张检测组件声明结束

    # 切换单张检测和批量检测
    batch_objs = {'folder': folder, 'start_detect': start_detect, 'title': title,
                  'results_list': results_list, 'result_select': result_select,     'category_select': category_select,
                  'confirm': confirm, 'prev_result': prev_result, 'next_result': next_result, 'report': report}
    single_objs = {'img': img, 'save': save, 'title': title}
    change = Button(interface_2_window, text="换为单张检测", width=10, height=1, font=('宋体', 10),
                    command=lambda: thread_it(change_type, change, batch_objs, single_objs, interface_2_window))
    change.place(relx=0.015, rely=0.022)

    ############################################################## 批量检测组件布置
    title.place(relx=0.32, rely=0.01)
    folder.place(relx=0.22, rely=0.12)
    # save_folder.place(relx=0.22, rely=0.24)
    start_detect.place(relx=0.22, rely=0.36)
    results_list.place(relx=0.72, rely=0.12)
    result_select.place(relx=0.42, rely=0.44)
    category_select.place(relx=0.42, rely=0.64)
    confirm.place(relx=0.72, rely=0.64)
    prev_result.place(relx=0.22, rely=0.84)
    next_result.place(relx=0.82, rely=0.84)
    report.place(relx=0.50, rely=0.80)
    ############################################################### 批量检测组件布置结束

    # 进入消息循环
    interface_2_window.mainloop()



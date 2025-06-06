"""
@author : hogan
time:
function: 性状检测功能界面及功能调用
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
import datetime

from functions.trait_detect import detect_2, detect_one_2, detect, detect_one
from functions.tools import thread_it, source_path, select_folder, save_img, clear_window, load_bg, load_image,\
    return_to_above, on_closing
from functions.review import set_combobox_placeholder,  refresh_listbox,  change_selection, \
     show_image, change_list, show_report_window, refresh_combobox


result_dict = {'file_dict': "", 'type': 0, 'res_img': None, 'name': "", 'save_path': ""} # 存储全局变量
pre_list = []
img_list = []
path_list = []
result_mode = "picture_selection"
review_index = -1

# 修改当前工作目录，使得资源文件可以被正确访问
cd = source_path('')
os.chdir(cd)

# 批量检测的函数
def detect_batch(indir, button, bar, window, results_list, result_select, type):
    global img_list
    global pre_list
    global path_list
    button['state'] = DISABLED
    button['text'] = '检测中'

    var = StringVar()
    l = Label(window, textvariable=var, bg="#f0f0f0", font=('宋体', 8), width=10, height=1)

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

    # 给模型输入输出路径等参数，开始检测
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")  # 系统能提供的device

    # cmd调用模型，传入--device，输入路径--source，输出文件夹集--project，具体的文件夹名--name
    var.set('检测中...')
    l.place(relx=0.60, rely=0.63)
    window.update()

    os_path = os.path.dirname(__file__)
    os_path = os.path.dirname(os_path)  # 返回项目目录，使用绝对路径
    if type == 'tusizi':
        weights = os.path.join(os_path, 'models/trait/best_hist.pt')
        classifier = os.path.join(os_path, 'models/trait/cls_hist_nom.pkl')
        pre_list, img_list, path_list = detect(type, bar, window, device, indir, weights, classifier)
        # pre_list = [inner_list for _, inner_list in pre_list_data]
    elif type == 'suanzaoren':
        weights = os.path.join(os_path, 'models/trait/macro_stage1.pt')
        pre_list, img_list, path_list = detect_2(type, bar, window, device, indir, weights)
    elif type == 'cheqianzi':
        weights = os.path.join(os_path, 'models/trait/stage1_best.pt')
        pre_list, img_list, path_list = detect_2(type, bar, window, device, indir, weights)
    else:
        print("type error! select the right type!!")

    print(pre_list)
    var.set('检测完成')
    l.place(relx=0.605, rely=0.624)
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
def select_img(var, label, window, button, button2, type):
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

    os_path = os.path.dirname(__file__)
    os_path = os.path.dirname(os_path)  # 返回项目目录，使用绝对路径
    if type == 'tusizi':
        weights = os.path.join(os_path, 'models/trait/best_hist.pt')
        classifier = os.path.join(os_path, 'models/trait/cls_hist_nom.pkl')
        res = detect_one(type, img, device, weights, classifier)
    elif type == 'suanzaoren':
        weights = os.path.join(os_path, 'models/trait/macro_stage1.pt')
        res = detect_one_2(type, img, device, weights)
    elif type == 'cheqianzi':
        weights = os.path.join(os_path, 'models/trait/stage1_best.pt')
        res = detect_one_2(type, img, device, weights)
    else:
        print("type error! select the right type!!")

    result_dict['res_img'] = res
    result_dict['name'] = file_path.split('/')[-1]

    window.update()

    button['state'] = NORMAL
    button['text'] = '选择图片'
    button2['state'] = NORMAL
    button2['text'] = '保存图片'

    # 展示结果
    res.tight_layout()
    res_path = os.path.join(os_path, 'single_result.png')
    res.savefig(res_path, dpi=300, bbox_inches='tight')
    print("single test result plot printed!")
    img = cv2.imread(res_path)
    img = cv2.resize(img, (850, 650))
    cv2.imwrite(res_path, img)

    global label_img
    img = Image.open(res_path)
    img.resize((300, 200))
    label_img = ImageTk.PhotoImage(img)
    l0 = Label(window,
            image=label_img,  # 标签的文字
            # bg="#f0f0f0",  # 标签背景颜色
            bg="#ffffff",  # 标签背景颜色
            # width=500, height=300  # 标签长宽
            )
    # l0.pack()

    l0.place(relx=0.07, rely=0.33)
    window.update()

# 修改单张和批量检测
def change_type(change, batch_objs, single_objs, window):
    ################################################# 重写转换函数，要正确摆放位置，并且要清除过时的label
    ori = window.winfo_children()

    if result_dict['type'] == 0: # 换为单张检测
        result_dict['type'] = 1

        # 删除批量检测各个组件
        for obj in ori:
            obj.place_forget()

        # 布置单张检测组件
        img = single_objs['img']
        save = single_objs['save']
        title = single_objs['title']
        return_prev = single_objs['return_prev']
        # l0 = single_objs['label']
        title.place(relx=0.25, rely=0.05)
        img.place(relx=0.20, rely=0.23)
        save.place(relx=0.62, rely=0.23)
        change.place(relx=0.04, rely=0.122)
        return_prev.place(relx=0.04, rely=0.042)
        # results_list.place(relx=0.18, rely=0.345)
        # result_select.place(relx=0.24, rely=0.68)
        # category_select.place(relx=0.64, rely=0.68)
        # confirm.place(relx=0.20, rely=0.77)
        # prev_result.place(relx=0.10, rely=0.74)
        # next_result.place(relx=0.855, rely=0.74)
        # report.place(relx=0.62, rely=0.77)

    else: # 换为批量检测
        result_dict['type'] = 0

        # 删除单张检测各个组件
        for obj in ori:
            obj.place_forget()

        # 布置批量检测各个组件
        folder = batch_objs['folder']
        # save_folder = batch_objs['save_folder']
        start_detect = batch_objs['start_detect']
        results_list = batch_objs['results_list']
        result_select = batch_objs['result_select']
        category_select = batch_objs['category_select']
        confirm = batch_objs['confirm']
        prev_result = batch_objs['prev_result']
        next_result = batch_objs['next_result']
        report = batch_objs['report']
        title = batch_objs['title']
        return_prev = batch_objs['return_prev']
        # l0 = batch_objs['label']
        return_prev.place(relx=0.04, rely=0.042)
        title.place(relx=0.25, rely=0.05)
        folder.place(relx=0.20, rely=0.23)
        change.place(relx=0.04, rely=0.122)
        # save_folder.place(relx=0.22, rely=0.24)
        start_detect.place(relx=0.62, rely=0.23)
        results_list.place(relx=0.18, rely=0.345)
        result_select.place(relx=0.24, rely=0.68)
        category_select.place(relx=0.64, rely=0.68)
        confirm.place(relx=0.20, rely=0.77)
        prev_result.place(relx=0.10, rely=0.74)
        next_result.place(relx=0.855, rely=0.74)
        report.place(relx=0.62, rely=0.77)

# 结果展示列表的选中事件：选择图片展示并查看细分结果;选择的同时设置好复查要用到的结果选择下拉栏内容
def result_list_event(event, window, img_list,  results_list, results_select):
    global result_mode
    global review_index
    if result_mode == "picture_selection":
        selection = event.widget.curselection()
        if not selection:
            return
        selected_index = int(selection[0])
        review_index = selected_index  # 保存图片索引
        show_image(img_list, window, selected_index)
        change_list(selected_index, results_list, results_select)
        result_mode = "result_review"
    elif result_mode == "result_review":
        selection = event.widget.curselection()
        if not selection:
            return
        selected_index = int(selection[0])
        review_index = selected_index
        if selected_index == 0:
            refresh_listbox(results_list, pre_list)
            result_mode = "picture_selection"
            results_select["values"] = ""
    else:
        return

# 人工结果复查，将结果列表中对应条目改为人工给出的类型
def refresh_result(category_select, results_list, result_select, window, result_mode):
    global pre_list
    var = StringVar()
    remind = Label(window, textvariable=var, width=15, height=2)
    if result_mode == 'result_review':
        selected_index = result_select.current()
        category_num = category_select.current()
        if selected_index == -1 or category_num == -1:  # 未选择结果或类别
            return
        selected_category = category_select.get()
        print("current review index:{}".format(review_index))
        print("selected index:{}".format(selected_index))
        print("prediction for review: {}".format(pre_list))
        current_result = pre_list[review_index]
        if type(current_result) != list:
            current_result = [current_result]

        if current_result[0] != "返回上一级":
            current_result.insert(0, "返回上一级")
        print("current chosen picture prediction: {}".format(current_result))
        current_result[selected_index + 1] = [selected_category, "人工复查"]
        print("renewed picture prediction: {}".format(current_result))
        refresh_listbox(results_list, current_result)
        refresh_combobox(result_select, current_result[1:])
        pre_list[review_index] = current_result[1:]
        var.set('所选结果已更新')
        remind.place(relx=0.47, rely=0.68)
        print("更新后的结果为：{}".format(pre_list))
    else:
        var.set('请选择细分结果')
        remind.place(relx=0.47, rely=0.68)

def show_interface(prev_window, info_batch, type):
    prev_window.withdraw()
    interface_window = tk.Toplevel(prev_window)
    interface_window.geometry('1000x750')
    interface_window.title(info_batch['title'])
    interface_window.resizable(False, False)  # 禁止最大化
    interface_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(prev_window, interface_window))

    # 设置背景图片
    os_path = os.path.dirname(__file__)
    os_path = os.path.dirname(os_path)  # 返回上级目录
    load_bg(interface_window, os_path, 'background/micro_detect.png')

    raw_path = info_batch['title_path']
    title_image = load_image(os_path, raw_path, 'title')
    title = Label(interface_window, image=title_image)

    #################################################### 图片读入
    start_detect_img = load_image(os_path, 'button/start_detect.png', 'detect')
    select_folder_img = load_image(os_path, 'button/select_folder.png', 'detect')
    confirm_img = load_image(os_path, 'button/confirm.png', 'detect')
    change_type_img = load_image(os_path, 'button/change_type.png', 'small')
    report_window_img = load_image(os_path, 'button/report_window.png', 'detect')
    return_img = load_image(os_path, 'button/return.png', 'small')
    select_img_img = load_image(os_path, 'button/select_img.png', 'detect')
    single_save_img = load_image(os_path, 'button/single_save.png', 'detect')
    ########################################################### 批量检测组件声明
    # lable，说明输入框作用
    var1 = StringVar()
    l1 = Label(interface_window,
               textvariable=var1,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 8),  # 字体和字体大小
               width=100, height=1  # 标签长宽
               )
    # 进度条
    # bar = tkinter.ttk.Progressbar(window)
    bar = Progressbar(interface_window)
    # 按钮，点击则开始一次批量检测
    start_detect = Button(interface_window, text='开始检测', width=200, height=50, image=start_detect_img,
                          command=lambda: thread_it(detect_batch, result_dict['file_dict'], start_detect, bar,
                                                    interface_window, results_list, result_select, type))

    # 按钮，选择待测图片所在文件夹
    folder = Button(interface_window, text="待测图片文件夹", width=200, height=50, image=select_folder_img,
                    command=lambda: thread_it(select_folder, var1, l1, folder, interface_window, result_dict))

    # 选择要修改的结果
    placeholder1 = "请选择要修正的结果"
    style1 = ttk.Style()
    style1.configure('Placeholder.TCombobox', foreground='gray')
    result_select = ttk.Combobox(interface_window, state="readonly")
    # result_select.bind("<<ComboboxSelected>>", lambda event, window=interface_window, combobox=result_select,
    #                                         listbox=results_list: result_select_event(event, window, combobox, listbox))

    result_select['values'] = ""
    result_select.pack(pady=10)

    # 结果展示列表
    results_list = tk.Listbox(interface_window, width=35, height=12)
    results_list.bind("<<ListboxSelect>>", lambda event, window=interface_window, results_list=results_list,
                      result_select=result_select:
                      result_list_event(event, window, img_list, results_list, result_select))
    results_list.pack(padx=10, pady=10)

    # 选择要修正为的类别
    placeholder2 = "请选择类别"
    style2 = ttk.Style()
    style2.configure('Placeholder.TCombobox', foreground='gray')
    category_select = Combobox(interface_window, state="readonly")
    category_select['values'] = info_batch['results_category']
    category_select.pack(pady=10)
    # 设置样式和占位符
    set_combobox_placeholder(result_select, placeholder1)
    set_combobox_placeholder(category_select, placeholder2)

    confirm = Button(interface_window, text='确认', width=200, height=50, font=('宋体', 10), image=confirm_img,
                     command=lambda: refresh_result(category_select, results_list, result_select, interface_window,
                                                    result_mode))
    max = len(pre_list)
    prev_result = Button(interface_window, text='上一条', width=8, command=lambda:
                         change_selection(result_select, 'prev', max))
    next_result = Button(interface_window, text='下一条', width=8, command=lambda:
                         change_selection(result_select, 'next', max))
    report = Button(interface_window, text='生成检测报告', width=200, height=50, image=report_window_img, command=lambda:
                    show_report_window(interface_window, report, result_dict, pre_list, path_list, img_list))
    #################################################### 批量检测组件声明结束

    ############################################################### 单张检测组件声明
    # 选择图片的标签
    var3 = StringVar()
    l3 = Label(interface_window,
               textvariable=var3,  # 标签的文字
               bg="#f0f0f0",  # 标签背景颜色
               font=('宋体', 10),  # 字体和字体大小
               width=90, height=1  # 标签长宽
               )

    # 保存图片按钮
    save = Button(interface_window, text="保存", width=200, height=50, image=single_save_img,
                  command=lambda: save_img(interface_window, save, result_dict))
    save['state'] = DISABLED

    # 选择图片并检测的按钮
    img = Button(interface_window, width=200, height=50, image=select_img_img,
                 command=lambda: thread_it(select_img, var3, l3, interface_window, img, save, type))
    ############################################################### 单张检测组件声明结束

    return_prev = Button(interface_window, width=120, height=35, image=return_img, command=lambda:
                         return_to_above(prev_window, interface_window))

    # 切换单张检测和批量检测
    batch_objs = {'folder': folder, 'start_detect': start_detect, 'return_prev': return_prev, 'title': title,
                  'results_list': results_list, 'result_select': result_select, 'category_select': category_select,
                  'confirm': confirm, 'prev_result': prev_result, 'next_result': next_result, 'report': report}
    single_objs = {'img': img, 'save': save, 'title': title, 'return_prev': return_prev }
    change = Button(interface_window, width=120, height=35, image=change_type_img,
                    command=lambda: thread_it(change_type, change, batch_objs, single_objs, interface_window))



    ############################################################## 批量检测组件布置
    title.place(relx=0.25, rely=0.05)
    folder.place(relx=0.20, rely=0.20)
    change.place(relx=0.04, rely=0.122)
    return_prev.place(relx=0.04, rely=0.042)
    # save_folder.place(relx=0.22, rely=0.24)
    start_detect.place(relx=0.62, rely=0.20)
    results_list.place(relx=0.18, rely=0.32)
    result_select.place(relx=0.24, rely=0.68)
    category_select.place(relx=0.64, rely=0.68)
    confirm.place(relx=0.20, rely=0.77)
    prev_result.place(relx=0.10, rely=0.74)
    next_result.place(relx=0.855, rely=0.74)
    report.place(relx=0.62, rely=0.77)
    ############################################################### 批量检测组件布置结束

    # 进入消息循环
    interface_window.mainloop()

def show_trait_window(prev_window):

    if prev_window.winfo_width() == 800:  # 这是从主界面进入
        # 创建药材选择主界面
        # window = prev_window
        prev_window.withdraw()
        window = tk.Toplevel(prev_window)
        window.title("中药材性状检测")
        window.geometry('1000x750')
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", lambda: on_closing(prev_window, window))
    else:  # 这是从子功能界面返回
        window = prev_window
        window.protocol("WM_DELETE_WINDOW", lambda: on_closing(prev_window, window))

    # 设置背景图片
    os_path = os.path.dirname(__file__)
    os_path = os.path.dirname(os_path) # 返回上级目录

    load_bg(window, os_path, 'background/detect.png')

    button1_image = load_image(os_path, 'button/trait_south.png', 'trait')
    button2_image = load_image(os_path, 'button/trait_suanzaoren.png', 'trait')
    button3_image = load_image(os_path, 'button/trait_cheqianzi.png', 'trait')
    return_image = load_image(os_path, 'button/return_to_main.png', 'small')

    category_1 = ('南方菟丝子-真品', '南方菟丝子-混淆品')
    category_2 = ('酸枣仁-真品', '酸枣仁-混淆品')
    category_3 = ('车前子-真品', '车前子-混淆品')
    tusizi_batch = {'title': "南方菟丝子及其混淆品智能检测", 'title_path': 'background/title/trait_tusizi.png',
                         'results_category': category_1}
    suanzaoren_batch = {'title': "酸枣仁及其混淆品智能检测", 'title_path': 'background/title/trait_suanzaoren.png',
                         'results_category': category_2}
    cheqianzi_batch = {'title': "车前子及其混淆品智能检测", 'title_path': 'background/title/trait_cheqianzi.png',
                         'results_category': category_3}
    # 创建按钮，指向界面
    button1 = tk.Button(window, image=button1_image, width=500, height=75,
                        command=lambda: show_interface(window, tusizi_batch, 'tusizi'))
    button1.pack(padx=20, pady=10)

    button2 = tk.Button(window, image=button2_image, width=500, height=75,
                        command=lambda: show_interface(window, suanzaoren_batch, 'suanzaoren'))
    button2.pack(padx=20, pady=10)

    button3 = tk.Button(window, image=button3_image, width=500, height=75,
                        command=lambda: show_interface(window, cheqianzi_batch, 'cheqianzi'))
    button3.pack(padx=20, pady=10)

    return_button = tk.Button(window, image=return_image, width=120, height=35,
                              command=lambda: return_to_above(prev_window, window))
    return_button.pack(padx=20, pady=10)

    button1.place(relx=0.28, rely=0.20)
    button2.place(relx=0.28, rely=0.40)
    button3.place(relx=0.28, rely=0.60)
    return_button.place(relx=0.04, rely=0.042)

    window.mainloop()

# def return_to_main(prev_window, window):
#     window.destroy()
#     prev_window.deiconify()
#
#
# def return_to_trait(window):
#     clear_window(window)
#     show_trait_window(window)


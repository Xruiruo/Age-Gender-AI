from PIL import Image, ImageTk #导入模块，用以读取图片
from tkinter import filedialog #文本对话选择框
import tkinter as tk           #用以制作简易界面
import cv2                     #图片裁剪时所用
#from gender_predict import PredictGender        #性别预测模型
from getface_from_webcam import Webcam    #自定义python文件，引入从Webcam上捕捉人脸的函数
#from age_preict import PredictAge
#from matplotlib import pyplot as plt   #测试绘制图片

# #裁剪图片
# def resize(w_box, h_box, pil_image):
#     w, h = pil_image.size
#     f1 = 1.0 * w_box / w
#     f2 = 1.0 * h_box / h
#     factor = min([f1, f2])
#     width = int(w * factor)
#     height = int(h * factor)
#     return pil_image.resize((width, height), Image.ANTIALIAS)
#
# #单击选择按钮时，回调的函数，用以选择图片和显示图片
# def select():
#     #定义全局变量
#     global lb_text      #文本显示标签，显示当前图片路径
#     global imLabel      #显示原图片的标签
#     global imLabel2     #显示剪切后留下的头像（人脸）
#     global the_image    #存储读取的图片对象
#     global the_image2   #存储裁剪后的头像（人脸）对象
#     global filename
#
#     filename = tk.filedialog.askopenfilename()      #获取图像路径
#
#     if filename != '':
#         name = filename.split('/')[-1]
#         lb_text.configure(text="您选择的文件是：\n" + name,justify='left',anchor = 'w')
#         the_image = cv2.imread(filename)
#         #测试
#         # print(the_image)
#         # plt.imshow(the_image)
#         # plt.show()
#         im_temp = Image.open(filename)
#         im_temp = resize(250, 250, im_temp)     #对图片进行裁剪
#         im_show1 = ImageTk.PhotoImage(im_temp)
#
#         the_image2 = getGenderModel.getFace(the_image)  #获取图像中的人脸
#         face = cv2.resize(the_image2, (250, 250,))
#         face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#
#         im_show2 = ImageTk.PhotoImage(face)
#
#         imLabel.configure(image=im_show1)
#         imLabel.image = im_show1        #为了使标签刷新后能成功显示图像，需要保持依赖
#         imLabel.pack()
#
#         imLabel2.configure(image=im_show2)
#         imLabel2.image = im_show2
#         imLabel2.pack()
#     else:
#         lb_text.configure(text="您没有选择任何文件")
#
# #预测性别
# def callback_predict_gender():
#     result = getGenderModel.getGenderForecast(the_image)
#     if result == 0:
#         var.set("女")
#     else:
#         var.set("男")
#
# #预测年龄
# def callback_predict_age(file_path):
#     print(file_path)
#     res = predict_age.predict_age(file_path)
#     var2.set(res)
#
# #保存图片
# def saveImg():
#     filename = tk.filedialog.asksaveasfilename()
#     print(the_image2)
#     cv2.imwrite(filename, the_image2)
#
# #从Webcam上获取图片
# def custom_image():
#     getTrainingData('getTrainData', 0, 'training_data_me/', 10)  # 注意这里的training_data_xx 文件夹就在程序工作目录下

class Screen():
    def __init__(self, predict_age, predict_gender):
        self.predict_age = predict_age
        self.predict_gender = predict_gender
        self.web_cam = Webcam(predict_age,predict_gender)
        self.the_image = 0
        self.the_image2 = []
        self.filename = ''

        self.root = tk.Tk()
        self.root.geometry("700x700+100+0")
        self.root.title("Gender&Age Forecast System")
        self.frame_left = tk.Frame(self.root)
        self.frame_top = tk.Frame(self.root)
        self.frame_bottom = tk.Frame(self.root)

        # 文本标签
        self.lb_text = tk.Label(self.frame_left, text='没有文件！！', font=('ariel', 10, 'bold'), bd=16, fg="steel blue")
        self.lb_text.pack()

        # 选择图像的按钮
        self.btn_select_img = tk.Button(self.frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10,
                                   text="选择图片",
                                   bg="powder blue", command=self.__select)
        self.btn_select_img.pack()

        self.var_gender = tk.StringVar()
        self.var_gender.set("???")

        # 预测性别的按钮
        self.btn_predict_gender = tk.Button(self.frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'),
                                       width=10, text="预测性别",
                                       bg="powder blue", command=self.__callback_predict_gender)
        self.btn_predict_gender.pack()

        self.text_gender = tk.Label(self.frame_left, textvariable=self.var_gender, font=('ariel', 16, 'bold'), bd=16, fg="steel blue")
        self.text_gender.pack()

        self.var_age = tk.StringVar()
        self.var_age.set("???")

        # 预测年龄的按钮
        self.btn_predict_age = tk.Button(self.frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'),
                                    width=10, text="预测年龄",
                                    bg="powder blue", command=lambda: self.__callback_predict_age(file_path=self.filename))

        self.btn_predict_age.pack()
        self.text2 = tk.Label(self.frame_left, textvariable=self.var_age, font=('ariel', 16, 'bold'), bd=16, fg="steel blue")
        self.text2.pack()

        # 保存图像的按钮
        self.btn_save_img = tk.Button(self.frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10,
                                 text="保存头像",
                                 bg="powder blue", command=self.__saveImg)
        self.btn_save_img.pack()

        # 自定义图像（webcam）
        self.btn_save_img = tk.Button(self.frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10,
                                 text="打开摄像头",
                                 bg="powder blue", command=self.__custom_image)
        self.btn_save_img.pack()

        #默认打开图片
        self.im = Image.open("testImg/nan2.jpg")
        self.im = self.__resize(250, 250, self.im)
        self.img = ImageTk.PhotoImage(self.im)
        # print(img)
        # 定义原图片标签
        self.im_title_original = tk.Label(self.frame_top, text='原图片', font=('ariel', 20, 'bold'), bd=16, fg="steel blue")
        self.im_title_original.pack()

        self.imLabel = tk.Label(self.frame_top, image=self.img, width=250, height=250, bg='#F0FFFF')
        self.imLabel.pack(side=tk.RIGHT)

        # 默认打开的图片及裁剪后的头像
        self.im_clip = Image.open("testImg/touxiang.png")
        self.im_clip = self.__resize(250, 250, self.im_clip)
        self.img_clip = ImageTk.PhotoImage(self.im_clip)
        # print(img)

        # 定义头像标签
        self.im_title_head = tk.Label(self.frame_bottom, text='头像', font=('ariel', 20, 'bold'), bd=16, fg="steel blue")
        self.im_title_head.pack()

        self.imLabel2 = tk.Label(self.frame_bottom, image=self.img_clip, width=250, height=250, bg="#F0FFFF")
        self.imLabel2.pack(side=tk.RIGHT)

        self.frame_left.pack(side=tk.LEFT)
        self.frame_top.pack(side=tk.TOP)
        self.frame_bottom.pack(padx=10, pady=10, side=tk.BOTTOM)

    def loop(self):
        self.root.mainloop()

    # 裁剪图片
    def __resize(self, w_box, h_box, pil_image):
        w, h = pil_image.size
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        return pil_image.resize((width, height), Image.ANTIALIAS)

    # 单击选择按钮时，回调的函数，用以选择图片和显示图片
    def __select(self):
        # 定义全局变量
        #global lb_text  # 文本显示标签，显示当前图片路径
        #global imLabel  # 显示原图片的标签
        #global imLabel2  # 显示剪切后留下的头像（人脸）
        #global the_image  # 存储读取的图片对象
        #global the_image2  # 存储裁剪后的头像（人脸）对象
        #global filename

        self.filename = tk.filedialog.askopenfilename()  # 获取图像路径

        if self.filename != '':
            name = self.filename.split('/')[-1]
            self.lb_text.configure(text="您选择的文件是：\n" + name, justify='left', anchor='w')
            self.the_image = cv2.imread(self.filename)
            # 测试
            # print(the_image)
            # plt.imshow(the_image)
            # plt.show()
            im_temp = Image.open(self.filename)
            im_temp = self.__resize(250, 250, im_temp)  # 对图片进行裁剪
            im_show1 = ImageTk.PhotoImage(im_temp)

            self.the_image2 = self.predict_gender.getFace(self.the_image)  # 获取图像中的人脸
            face = cv2.resize(self.the_image2, (250, 250,))
            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            im_show2 = ImageTk.PhotoImage(face)

            self.imLabel.configure(image=im_show1)
            self.imLabel.image = im_show1  # 为了使标签刷新后能成功显示图像，需要保持依赖
            self.imLabel.pack()

            self.imLabel2.configure(image=im_show2)
            self.imLabel2.image = im_show2
            self.imLabel2.pack()
            self.var_age.set('???')
            self.var_gender.set('???')
        else:
            self.lb_text.configure(text="您没有选择任何文件")

    # 预测性别
    def __callback_predict_gender(self):
        print(self.the_image)
        result = self.predict_gender.getGenderForecast(self.the_image)
        if result == 0:
            self.var_gender.set("女")
        else:
            self.var_gender.set("男")

    # 预测年龄
    def __callback_predict_age(self, file_path):
        print(file_path)
        res = self.predict_age.predict_age(file_path)
        self.var_age.set(res)

    # 保存图片
    def __saveImg(self):
        filename = tk.filedialog.asksaveasfilename()
        print(self.the_image2)
        cv2.imwrite(filename, self.the_image2)

    # 从Webcam上获取图片
    def __custom_image(self):
        self.web_cam.get_face_from_webcam()
        #getTrainingData('getTrainData', 0, 'training_data_me/', 10)  # 注意这里的training_data_xx 文件夹就在程序工作目录下
# if __name__ == '__main__':
#     predict_age = PredictAge()
#     s = Screen(predict_age)
#     s.start()
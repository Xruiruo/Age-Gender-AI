import cv2

# def getTrainingData(window_name, camera_id, path_name, max_num):  # path_name是图片存储目录，max_num是需要捕捉的图片数量
#     cv2.namedWindow(window_name)  # 创建窗口
#     cap = cv2.VideoCapture(camera_id)  # 打开摄像头
#     classifier = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt2.xml')  # 加载分类器
#     color = (0, 255, 0)  # 人脸矩形框的颜色
#     #num = 0  # 记录存储的图片数量
#
#     while cap.isOpened():
#         ok, frame = cap.read()
#         if not ok:
#             break
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
#         face_rects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#
#         if len(face_rects) > 0:
#             for face_rect in face_rects:
#                 x, y, w, h = face_rect
#                 # 捕捉到的图片的名字，这里用到了格式化字符串的输出
#                 image_name = '%s%d.jpg' % (path_name, num)  # 注意这里图片名一定要加上扩展名，否则后面imwrite的时候会报错：could not find a writer for the specified extension in function cv::imwrite_ 参考：https://stackoverflow.com/questions/9868963/cvimwrite-could-not-find-a-writer-for-the-specified-extension
#                 image = frame[y:y+h, x:x+w]  # 将当前帧含人脸部分保存为图片，注意这里存的还是彩色图片，前面检测时灰度化是为了降低计算量；这里访问的是从y位开始到y+h-1位
#                 cv2.imwrite(image_name, image)
#
#                 num += 1
#                 # 超过指定最大保存数量则退出循环
#                 if num > max_num:
#                     break
#
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)  # 画出矩形框
#                 font = cv2.FONT_HERSHEY_SIMPLEX  # 获取内置字体
#                 cv2.putText(frame, ('%d'%num), (x+30, y+30), font, 1, (255, 0, 255), 4) # 调用函数，对人脸坐标位置，添加一个(x+30,y+30）的矩形框用于显示当前捕捉到了多少人脸图片
#         if num > max_num:
#             break
#         cv2.imshow(window_name, frame)
#         c = cv2.waitKey(10)
#         if c & 0xFF == ord('q'):
#             break
#
#     cap.release()#释放摄像头并销毁所有窗口
#     cv2.destroyAllWindows()
#     print('Finished.')

class Webcam():
    def __init__(self, predict_age, predict_gender):
        self.predict_age = predict_age
        self.predict_gender = predict_gender
        self.window_name = 'Age&Gender AI'
        self.camera_id = 0

    def get_face_from_webcam(self):
        cv2.namedWindow(self.window_name)  # 创建窗口
        cap = cv2.VideoCapture(self.camera_id)  # 打开摄像头
        classifier = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt2.xml')  # 加载分类器
        color = (0, 255, 0)  # 人脸矩形框的颜色
        num = 0  # 记录存储的图片数量

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
            face_rects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))

            if len(face_rects) > 0:
                for face_rect in face_rects:
                    x, y, w, h = face_rect
                    # 捕捉到的图片的名字，这里用到了格式化字符串的输出
                    #image_name = '%s%d.jpg' % (path_name,
                    #                          num)  # 注意这里图片名一定要加上扩展名，否则后面imwrite的时候会报错：could not find a writer for the specified extension in function cv::imwrite_ 参考：https://stackoverflow.com/questions/9868963/cvimwrite-could-not-find-a-writer-for-the-specified-extension
                    image = frame[y:y + h, x:x + w]  # 将当前帧含人脸部分保存为图片，注意这里存的还是彩色图片，前面检测时灰度化是为了降低计算量；这里访问的是从y位开始到y+h-1位
                    #cv2.imwrite(image_name, image)
                    res_gender = self.predict_gender.getGenderForecast(image)
                    res_age = self.predict_age.predict_age(img=image)
                    if res_gender == 0:
                        res_gender = 'f'
                    else:
                        res_gender = 'm'
                    #num += 1
                    # 超过指定最大保存数量则退出循环
                    #if num > max_num:
                        #break
                    text_age_gender = res_gender+","+str(res_age)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # 画出矩形框
                    font = cv2.FONT_HERSHEY_SIMPLEX  # 获取内置字体
                    cv2.putText(frame, text_age_gender, (x + 30, y + 30), font, 1, (255, 0, 255),
                                4)  # 调用函数，对人脸坐标位置，添加一个(x+30,y+30）的矩形框用于显示当前捕捉到了多少人脸图片
            #if num > max_num:
                #break
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1)
            if key == 27:  #ESC
                break

        cap.release()  # 释放摄像头并销毁所有窗口
        cv2.destroyAllWindows()
        print('Finished.')
#主函数
#if __name__ =='__main__':
    #print ('catching your face and writting into disk...')
    #getTrainingData('getTrainData',0,'training_data_me/',100) # 注意这里的training_data_xx 文件夹就在程序工作目录下
    #webcam = Webcam()
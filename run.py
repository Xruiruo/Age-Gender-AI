from gui import Screen
from age_preict import PredictAge
from gender_predict import PredictGender

if __name__ == '__main__':
    #生成年龄预测模型实例
    predict_age = PredictAge()
    predict_gender = PredictGender()
    #生成绘制的界面实例
    screen = Screen(predict_age, predict_gender)
    #屏幕循环
    screen.loop()